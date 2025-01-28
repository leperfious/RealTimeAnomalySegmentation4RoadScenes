import os
import argparse
import numpy as np
from PIL import Image

num_classes = 20 # 19 classes + 1 void

# calculate class pixel freq
def calculate_class_frequencies(gtFine_dir, num_classes):
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    # Iterate through all ground truth images
    for root, _, files in os.walk(gtFine_dir):
        for file in files:
            if file.endswith('_labelTrainIds.png'):
                # load the ground truth image
                gt_image = np.array(Image.open(os.path.join(root, file)))

                # count pixels for each classs
                for class_id in range(num_classes):
                    class_pixel_counts[class_id] += np.sum(gt_image == class_id)

                # total pixel count
                total_pixels += gt_image.size

    return class_pixel_counts, total_pixels

# calculate for each
def calculate_weights(class_pixel_counts, total_pixels, architecture):
    class_weights = None

    if architecture == 'ERFNet_encoder':
        # used inverse class frequency for encoder
        class_weights = total_pixels / (class_pixel_counts + 1e-6)
    elif architecture == 'ERFNet_decoder':
        # used median frequency balancing for decoder
        median_frequency = np.median(class_pixel_counts[class_pixel_counts > 0] / total_pixels)
        class_weights = median_frequency / ((class_pixel_counts / total_pixels) + 1e-6)
    elif architecture == 'ENet':
        # used logarithmic weighting for ENet
        class_proportions = class_pixel_counts / total_pixels
        class_weights = 1 / (np.log(1.02 + class_proportions) + 1e-6)
    elif architecture == 'BiSeNet':
        # Used inverse square root of frequency for BiSeNet, because of using our loss function, i use weights.
        class_proportions = class_pixel_counts / total_pixels
        class_weights = 1 / (np.sqrt(class_proportions) + 1e-6)
    
    # used normalization, because i was getting very silly results.
    class_weights = 1 + 9 * (np.log1p(class_weights) / np.log1p(np.max(class_weights)))
    
    return class_weights


def main():
    parser = argparse.ArgumentParser(description="Calculate class weights for semantic segmentation.")
    parser.add_argument(
        '--gtFine_dir', # content//
        type=str, 
        required=True, 
        help="Path to the gtFine directory containing the 'train' subdirectory."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='weights.txt', 
        help="Path to save the output weights file."
    )
    args = parser.parse_args()

    gtFine_dir = os.path.join(args.gtFine_dir, 'train')

    print("Calculating class frequencies...")
    class_pixel_counts, total_pixels = calculate_class_frequencies(gtFine_dir, num_classes)

    architectures = ['ERFNet_encoder', 'ERFNet_decoder', 'ENet', 'BiSeNet']

    # calculate and save
    with open(args.output_file, 'w') as f:
        for architecture in architectures:
            print(f"Calculating weights for {architecture}...")
            class_weights = calculate_weights(class_pixel_counts, total_pixels, architecture)

            class_weights = [float(f"{w:.6f}") for w in class_weights]

            # save
            f.write(f"{architecture} weights:\n")
            f.write(','.join(map(str, class_weights)) + '\n')

    print(f"Class weights saved to {args.output_file}")

if __name__ == '__main__':
    main()
    #
