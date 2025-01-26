import os
import torch
from glob import glob
from torchvision.transforms import ToTensor
from PIL import Image

NUM_CLASSES = 20  # 19 classes + void

def calculate_weights(gt_dir, save_path="/content/class_weights.txt"):
    """
    Calculate class weights based on the label distribution in gtFine.
    Args:
        gt_dir (str): Path to the gtFine/train directory.
        save_path (str): Path to save the calculated weights as a .txt file.
    """
    label_counts = torch.zeros(NUM_CLASSES)

    # Get all label images
    label_files = glob(os.path.join(gt_dir, "**/*_trainId.png"), recursive=True)

    # Count pixel occurrences for each class
    for label_file in label_files:
        label_img = Image.open(label_file)
        label_tensor = torch.tensor(label_img, dtype=torch.long)  # Convert to tensor
        label_counts += torch.bincount(label_tensor.flatten(), minlength=NUM_CLASSES)

    # Safeguard against zero pixels
    print(f"Pixel counts for each class: {label_counts.tolist()}")
    total_samples = label_counts.sum()
    weights = torch.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if label_counts[i] > 0:
            weights[i] = 1 / (label_counts[i] / total_samples)
        else:
            weights[i] = 0.0  # Assign 0 weight to classes with no pixels

    # Write weights to a .txt file
    with open(save_path, "w") as f:
        f.write("Class Weights:\n")
        for i, weight in enumerate(weights):
            f.write(f"Class {i}: {weight.item()}\n")
    print(f"Class weights saved to {save_path}")

    return weights

def main():
    gt_train_dir = "/content/datasets/cityscapes/gtFine/train"  # Path to gtFine/train
    assert os.path.exists(gt_train_dir), "Error: gtFine/train directory not found!"

    # Calculate and save weights
    calculate_weights(gt_train_dir, save_path="/content/class_weights.txt")

if __name__ == '__main__':
    main()