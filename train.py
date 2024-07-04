import os
from datetime import datetime
import logging
import argparse

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import torch.nn.functional as F

from schedulefree import AdamWScheduleFree

from dataloader import SquareImageDataset, transform, custom_collate
from vae import AutoEncoder
from utils import load_config, store_safetensors


def prime_optimizer(model, lr, warmup_steps):
    # defazio's optimizer
    optimizer_state = AdamWScheduleFree(
        [
            # shouldn't use list comprehension, but this is basically filtering the bias and norm layer to not use weight decay
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if any([excluded in name for excluded in ["bias", "norm"]])
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if not any([excluded in name for excluded in ["bias", "norm"]])
                ]
            },
        ],
        weight_decay=1e-2,  # bog standard no need to change this
        betas=(0.9, 0.999),  # bog standard no need to change this
        lr=lr,
        warmup_steps=warmup_steps,  # settable because the model cannot be initialized with identity
    )
    return optimizer_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load configuration from a JSON file.')
    parser.add_argument('model_config', type=str, help='Path to modeling config file')
    parser.add_argument('training_config', type=str, help='Path to training config file')
    args = parser.parse_args()

    training_config = load_config(args.training_config)
    # uncomment this for debugging bypass
    # training_config = {
    #     "image_path": "train_images",
    #     "training_preview_path": "preview",
    #     "master_seed": 0,
    #     "step_counter_start": 0,
    #     "epoch": 10,
    #     "lr": 1e-3,
    #     "warmup": 10,
    #     "batch_size": 32,
    #     "grad_accum_count": 1,
    #     "val_percentage": 0.1,
    #     "check_point_every": 1000,
    #     "num_of_dataloader_workers": 2,
    #     "checkpoint_path": "path",
    #     "wandb_project_name": None,
    #     "run_name": "vae_training",
    #     "logging_file": "training.log",
    #     "device": "cuda:0",
    # }

    # init master seed for repro
    torch.manual_seed(training_config["master_seed"])

    # init wandb
    if training_config["wandb_project_name"] is not None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project=training_config["wandb_project_name"],
            name=f"{current_datetime}_{training_config['run_name']}",
        )

    # ensure ckpt folder exist
    if not os.path.exists(training_config["checkpoint_path"]):
        os.makedirs(training_config["checkpoint_path"])

    # ensure preview path exist
    if not os.path.exists(training_config["training_preview_path"]):
        os.makedirs(training_config["training_preview_path"])

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=training_config["logging_file"],
    )

    # init model
    ae_conf = load_config(args.model_config)
    model = AutoEncoder(**ae_conf)
    model.to(training_config["device"])
    optimizer = prime_optimizer(model, training_config["lr"], training_config["warmup"])

    # random tensor for testing
    # x = torch.randn(16, 3, 256, 256).to("cuda:0")

    # training counter state
    if training_config["step_counter_start"] > 0:
        counter = training_config["step_counter_start"]
    else:
        counter = 0

    # init loss value
    loss_counter = torch.tensor(0.0, dtype=torch.float32).to(training_config["device"])
    loss_l1_counter = torch.tensor(0.0, dtype=torch.float32).to(
        training_config["device"]
    )
    loss_l2_counter = torch.tensor(0.0, dtype=torch.float32).to(
        training_config["device"]
    )

    # init dataset
    dataset = SquareImageDataset(
        root_dir=training_config["image_path"],
        transform=transform,
        seed=training_config["master_seed"],
    )

    # split train validation set
    train_set_size = int(len(dataset) * (1 - training_config["val_percentage"]))
    val_set_size = len(dataset) - train_set_size
    dataset, val_dataset = random_split(dataset, [train_set_size, val_set_size])

    # init validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_of_dataloader_workers"],
        pin_memory=True,
        collate_fn=custom_collate,
    )
    val_iterator = iter(val_dataloader)

    for epoch in range(training_config["epoch"]):
        # init dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_of_dataloader_workers"],
            pin_memory=True,
            collate_fn=custom_collate,
        )
        # pretty loading bar
        progress_bar = tqdm(
            total=len(dataloader) // training_config["batch_size"], smoothing=0.5, dynamic_ncols=True
        )

        for step, batch in enumerate(dataloader):
            # safe catch
            if batch is None:
                logging.error(
                    f"batch error at {step} seed {training_config['master_seed']}"
                )
                progress_bar.update(1)
                counter += 1
                continue

            # move the batch to GPUs
            # batch_cuda = optree.tree_map(
            #     lambda x: x.squeeze(0).to(DEVICE, non_blocking=True), batch
            # )
            batch_cuda = batch.to(training_config["device"], non_blocking=True)

            # compute loss, l1, l2, latent, x
            loss, l1, l2, latent, x = model.loss_and_grad(
                batch_cuda,
                l1=0,
                checkpoint=True,
                grad_accum_steps=training_config["grad_accum_count"],
            )

            # optimizer step
            if (step + 1) % training_config["grad_accum_count"] == 0:

                optimizer.step()
                optimizer.zero_grad()
                loss_counter = loss / training_config["grad_accum_count"]
                loss_l1_counter = l1 / training_config["grad_accum_count"]
                loss_l2_counter = l2 / training_config["grad_accum_count"]

            # save model periodically
            if counter % training_config["check_point_every"] == 0:
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                checkpoint_folder = os.path.join(
                    training_config["checkpoint_path"],
                    f"{counter}_{current_datetime}_{training_config['run_name']}",
                )
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                store_safetensors(
                    model.state_dict(),
                    os.path.join(checkpoint_folder, "model.safetensors"),
                )
                # to lazy to flatten and save it as safetensors
                # only useful for training anyway
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, "optim_state.pth"))
                logging.info(f"Saved checkpoint at {checkpoint_folder}")

            # log wandb
            if counter % training_config["grad_accum_count"] * 10 == 0:
                if training_config["wandb_project_name"] is not None:
                    wandb.log(
                        {
                            "loss": loss_counter,
                            "loss_l1": loss_l1_counter,
                            "loss_l2": loss_l2_counter,
                            "lr": training_config["lr"]
                        },
                        step=counter,
                    )
                progress_bar.set_description(
                    f"Processing epoch {epoch}/{training_config['epoch']} || loss:{loss_counter}"
                )

            if counter % (training_config["batch_size"] * training_config["grad_accum_count"])==0:
                with torch.no_grad():
                    try:
                        while True:
                            val_batch = next(val_iterator)
                            if val_batch is not None:
                                break

                    except StopIteration:
                        # StopIteration is thrown if dataset ends
                        # reinitialize data loader
                        val_iterator = iter(val_dataloader)
                        while True:
                            val_batch = next(val_iterator)
                            if val_batch is not None:
                                break
                    finally:
                        # val_batch_cuda = optree.tree_map(
                        #     lambda x: x.squeeze(0).to(DEVICE, non_blocking=True),
                        #     val_batch,
                        # )
                        val_batch_cuda = val_batch.to(
                            training_config["device"], non_blocking=True
                        )
                        # compute loss, l1, l2, latent, x
                        val_loss, _, _, _, _ = model.loss_and_grad(
                            val_batch_cuda,
                            l1=0,
                            checkpoint=True,
                            grad_accum_steps=training_config["grad_accum_count"],
                            compute_grad=False,
                        )
                        if training_config["wandb_project_name"] is not None:
                            wandb.log({"val_loss": val_loss}, step=counter)

                        # preview
                        preview_out = model(val_batch_cuda)
                        save_image(
                            torch.cat([preview_out[:4], val_batch_cuda[:4]]),
                            f"{training_config['training_preview_path']}/{step:08}.png",
                            nrow=4,
                            normalize=True,
                        )

            progress_bar.update(1)
            counter += 1

        torch.manual_seed(epoch)

    print()
