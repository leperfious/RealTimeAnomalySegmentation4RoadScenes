import os
import torch
from glob import glob
from torchvision.transforms import ToTensor
from PIL import Image

NUM_CLASSES = 20  # 19 classes + void

def calculate_weights(gt_dir, method="inverse_frequency", save_path=None):
    """
    Calculate class weights based on the label distribution in gtFine.
    Args:
        gt_dir (str): Path to the gtFine/train directory.
        method (str): Weight calculation method. Options are:
                      - "inverse_frequency" (default for ERFNet encoder/decoder)
                      - "enet" (logarithmic formula for ENet)
        save_path (str): Path to save the calculated weights. If None, weights are not saved.
    Returns:
        weights (torch.Tensor): Calculated class weights.
    """
    label_counts = torch.zeros(NUM_CLASSES)

    # Get all label images
    label_files = glob(os.path.join(gt_dir, "**/*_labelTrainIds.png"), recursive=True)

    # Count pixel occurrences for each class
    for label_file in label_files:
        label_img = Image.open(label_file)
        label_tensor = torch.tensor(label_img, dtype=torch.long)  # Convert to tensor
        label_counts += torch.bincount(label_tensor.flatten(), minlength=NUM_CLASSES)

    # Safeguard against zero pixels
    print(f"Pixel counts for each class: {label_counts.tolist()}")
    total_samples = label_counts.sum()

    # Initialize weights
    weights = torch.zeros(NUM_CLASSES)

    if method == "inverse_frequency":
        # Inverse frequency: 1 / (class frequency / total samples)
        for i in range(NUM_CLASSES):
            if label_counts[i] > 0:
                weights[i] = 1 / (label_counts[i] / total_samples)
            else:
                weights[i] = 0.0  # Assign 0 weight to classes with no pixels
    elif method == "enet":
        # ENet's logarithmic weighting formula: w_c = ln(k + 1/f_c)
        k = 1.02  # Slightly higher than 1 to avoid extreme scaling
        for i in range(NUM_CLASSES):
            if label_counts[i] > 0:
                weights[i] = torch.log(k + (1 / (label_counts[i] / total_samples)))
            else:
                weights[i] = 0.0  # Assign 0 weight to classes with no pixels
    else:
        raise ValueError(f"Unsupported weight calculation method: {method}")

    # Save weights to a .txt file (optional)
    if save_path:
        with open(save_path, "w") as f:
            f.write(f"Class Weights ({method} method):\n")
            for i, weight in enumerate(weights):
                f.write(f"Class {i}: {weight.item():.6f}\n")
        print(f"Class weights saved to {save_path}")

    return weights


def main():
    gt_train_dir = "/content/datasets/cityscapes/gtFine/train"  # Path to gtFine/train
    assert os.path.exists(gt_train_dir), "Error: gtFine/train directory not found!"

    # Calculate and save weights for ERFNet Encoder
    erfnet_encoder_weights = calculate_weights(
        gt_train_dir, method="inverse_frequency", save_path="/content/erfnet_encoder_weights.txt"
    )
    torch.save(erfnet_encoder_weights, "/content/erfnet_encoder_weights.pth")
    print("ERFNet Encoder weights:", erfnet_encoder_weights.tolist())

    # Calculate and save weights for ERFNet Decoder
    erfnet_decoder_weights = calculate_weights(
        gt_train_dir, method="inverse_frequency", save_path="/content/erfnet_decoder_weights.txt"
    )
    torch.save(erfnet_decoder_weights, "/content/erfnet_decoder_weights.pth")
    print("ERFNet Decoder weights:", erfnet_decoder_weights.tolist())

    # Calculate and save weights for ENet
    enet_weights = calculate_weights(
        gt_train_dir, method="enet", save_path="/content/enet_weights.txt"
    )
    torch.save(enet_weights, "/content/enet_weights.pth")
    print("ENet weights:", enet_weights.tolist())


if __name__ == '__main__':
    main()