import os
import numpy as np
from PIL import Image

# Number of classes (19 known + 1 void)
num_classes = 20

# Function to calculate class pixel frequencies
def calculate_class_frequencies(gtFine_dir, num_classes):
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    # Iterate through all ground truth images
    for root, _, files in os.walk(gtFine_dir):
        for file in files:
            if file.endswith('_labelTrainIds.png'):
                # Load the ground truth image
                gt_image = np.array(Image.open(os.path.join(root, file)))

                # Count pixels for each class
                for class_id in range(num_classes):
                    class_pixel_counts[class_id] += np.sum(gt_image == class_id)

                # Update total pixel count
                total_pixels += gt_image.size

    return class_pixel_counts, total_pixels

# Function to calculate weights for different architectures
def calculate_weights(class_pixel_counts, total_pixels, architecture):
    class_weights = None

    if architecture == 'ERFNet_encoder':
        # Use inverse class frequency for encoder
        class_weights = total_pixels / (class_pixel_counts + 1e-6)
    elif architecture == 'ERFNet_decoder':
        # Use median frequency balancing for decoder
        median_frequency = np.median(class_pixel_counts[class_pixel_counts > 0] / total_pixels)
        class_weights = median_frequency / (class_pixel_counts / total_pixels + 1e-6)
    elif architecture == 'ENet':
        # Use logarithmic weighting for ENet
        class_proportions = class_pixel_counts / total_pixels
        class_weights = 1 / (np.log(1.02 + class_proportions) + 1e-6)
    elif architecture == 'BiSeNet':
        # Use inverse square root of frequency for BiSeNet
        class_proportions = class_pixel_counts / total_pixels
        class_weights = 1 / (np.sqrt(class_proportions) + 1e-6)

    # Normalize weights to avoid excessively large values
    class_weights = class_weights / np.sum(class_weights)
    return class_weights

# Main script
def main():
    # Define paths to Cityscapes dataset directories
    cityscapes_dir = '/content/datasets/cityscapes'
    gtFine_dir = os.path.join(cityscapes_dir, 'gtFine/train')

    # Output file
    output_file = 'weights.txt'

    print("Calculating class frequencies...")
    class_pixel_counts, total_pixels = calculate_class_frequencies(gtFine_dir, num_classes)

    architectures = ['ERFNet_encoder', 'ERFNet_decoder', 'ENet', 'BiSeNet']

    # Calculate and save weights for each architecture
    with open(output_file, 'w') as f:
        for architecture in architectures:
            print(f"Calculating weights for {architecture}...")
            class_weights = calculate_weights(class_pixel_counts, total_pixels, architecture)

            # Save weights to file
            f.write(f"{architecture} weights:\n")
            f.write(','.join([f"{w:.6f}" for w in class_weights]) + '\n')

    print(f"Class weights saved to {output_file}")

if __name__ == '__main__':
    main()
