import os
import json
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from glob import glob

NUM_CLASSES = 20  # 19 classes + void

def calculate_weights(gt_dir, save_path="/content/class_weights.json"):
    """
    Calculate class weights based on the label distribution in gtFine.
    Args:
        gt_dir (str): Path to the gtFine/train directory.
        save_path (str): Path to save the calculated weights.
    """
    label_counts = torch.zeros(NUM_CLASSES)

    # Get all label images
    label_files = glob(os.path.join(gt_dir, "**/*_trainId.png"), recursive=True)

    # Count pixel occurrences for each class
    for label_file in label_files:
        label_img = Image.open(label_file)
        label_tensor = torch.tensor(label_img, dtype=torch.long)  # Convert to tensor
        label_counts += torch.bincount(label_tensor.flatten(), minlength=NUM_CLASSES)

    # Calculate weights
    total_samples = label_counts.sum()
    weights = 1 / (label_counts / total_samples)

    # Save weights to JSON
    with open(save_path, "w") as f:
        json.dump(weights.tolist(), f)
    print(f"Class weights saved to {save_path}")

    return weights

def main():
    gt_train_dir = "/content/datasets/cityscapes/gtFine/train"  # Path to gtFine/train
    assert os.path.exists(gt_train_dir), "Error: gtFine/train directory not found!"

    # Calculate and save weights
    calculate_weights(gt_train_dir, save_path="/content/class_weights.json")

if __name__ == '__main__':
    main()