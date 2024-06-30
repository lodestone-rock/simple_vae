import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

# Custom Dataset class with error handling
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None


class RandomCropAndRotate:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        width, height = image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(f"Image size ({width}, {height}) is smaller than crop size ({self.crop_size}, {self.crop_size})")
        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        cropped_image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        rotate_angle = random.choice([0, 90, 180, 270])
        rotated_image = cropped_image.rotate(rotate_angle)
        return rotated_image

# Define your transformations
transform = transforms.Compose([
    RandomCropAndRotate(crop_size=256),
    transforms.ToTensor(),
])


# Custom collate function to handle None values (failed image loads)
def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Remove None values
    return torch.utils.data.dataloader.default_collate(batch)


# Example usage:
if __name__ == "__main__":
    # Create your dataset instance
    dataset = CustomDataset(root_dir='/media/lodestone/bulk_storage_2/datasets/imagenet/train_images', transform=transform)

    # Create DataLoader instance with custom collate_fn, pin_memory and non-blocking
    batch_size = 32  # Adjust as needed
    num_workers = 4  # Adjust based on your system's capabilities

    # Set pin_memory=True and non_blocking=True for faster GPU data transfer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers, pin_memory=True)

    # Example usage:
    for batch in dataloader:
        # batch is a tensor of shape (batch_size, channels, height, width)
        # Use batch for training or other purposes
        print(f"Loaded batch with shape: {batch.shape}")
