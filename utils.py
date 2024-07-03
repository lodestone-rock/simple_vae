import json
from safetensors import safe_open
from safetensors.torch import save_file


def load_safetensors(safetensor_path):
    tensors = {}
    with safe_open(safetensor_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return tensors


store_safetensors = save_file


def read_json_as_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def dump_dict_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def count_trainable_params(model):
    total_trainable = 0
    total_params = 0
    for n, p in model.named_parameters():
        params_count = p.numel()
        if p.requires_grad:
            total_trainable += params_count
        total_params += params_count
    return total_trainable, total_params


def save_list_as_jsonl(data, filename):
    """
    Save a list of dictionaries as a JSONL file.

    Args:
    - data (list): List of dictionaries to be saved.
    - filename (str): The name of the file to save the data.
    """
    with open(filename, "w") as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + "\n")


def load_jsonl_as_list(filename):
    """
    Load data from a JSONL file into a list of dictionaries.

    Args:
    - filename (str): The name of the file to load the data from.

    Returns:
    - list: List of dictionaries loaded from the JSONL file.
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


load_config = read_json_as_dict
