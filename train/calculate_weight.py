import os
import json
import torch
from dataset import cityscapes
from torchvision.transforms import Resize, ToTensor
from transform import Relabel, ToLabel

NUM_CLASSES = 20  # 19 classes + void

# Augmentations for input and target images
class MyCoTransform(object):
    def __init__(self, height=512):
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

# Function to calculate class weights
def calculate_weights(dataset, save_path="/content/class_weights.json"):
    label_counts = torch.zeros(NUM_CLASSES)

    for _, labels in dataset:
        label_counts += torch.bincount(labels.flatten(), minlength=NUM_CLASSES)

    total_samples = label_counts.sum()
    weights = 1 / (label_counts / total_samples)

    # Save weights to a JSON file
    with open(save_path, "w") as f:
        json.dump(weights.tolist(), f)
    print(f"Class weights saved to {save_path}")

    return weights

def main():
    datadir = "/content/datasets/cityscapes"  # Path to Cityscapes dataset
    assert os.path.exists(datadir), "Error: Dataset directory not found!"

    # Initialize dataset
    co_transform = MyCoTransform(height=512)
    dataset_train = cityscapes(datadir, co_transform, split="train")

    # Calculate and save weights
    calculate_weights(dataset_train, save_path="/content/class_weights.json")

if __name__ == '__main__':
    main()