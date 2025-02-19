# Code to calculate IoU (mean and per-class) in a dataset
# FEB 2025
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from argparse import ArgumentParser
from PIL import Image

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from enet import ENet
from bisenet import BiSeNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

# Set number of channels and classes
NUM_CHANNELS = 3
NUM_CLASSES = 20  # 19 known classes + void

# Image transformation for Cityscapes dataset
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # Void class mapped to 19
])

# Bisenet-specific transforms (matches training preprocessing)
input_transform_cityscapes_bisenet = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
])

target_transform_cityscapes_bisenet = Compose([
    Resize((512, 1024), Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),
])

# Custom function to handle missing/unexpected keys when loading weights
def load_bisenet_state_dict(model, state_dict):
    """ Load BiSeNet weights while handling mismatched keys. """
    
    new_state_dict = {}

    for k, v in state_dict.items():
        # Remove "module." prefix if present
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"🔎 Missing keys: {missing_keys}")
    print(f"🔎 Unexpected keys: {unexpected_keys}")

    return model

def main(args):

    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print(f"🔹 Loading model: {modelpath}")
    print(f"🔹 Loading weights: {weightspath}")

    # Adjust transforms for BiSeNet
    global input_transform_cityscapes, target_transform_cityscapes

    if args.model == 'ENet':
        model = ENet(NUM_CLASSES)
    elif args.model == 'BiSeNet':
        model = BiSeNet(NUM_CLASSES)
        input_transform_cityscapes = input_transform_cityscapes_bisenet
        target_transform_cityscapes = target_transform_cityscapes_bisenet
    else:
        model = ERFNet(NUM_CLASSES)

    # Move model to GPU if available
    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    # Load the model weights with `weights_only=True` for security
    state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True)

    if args.model == 'BiSeNet':
        model = load_bisenet_state_dict(model, state_dict)
    elif args.model == 'ENet':
        state_dict = {k if k.startswith("module.") else "module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    print("✅ Model and weights LOADED SUCCESSFULLY.")

    model.eval()  # Set model to evaluation mode

    if not os.path.exists(args.datadir):
        print("❌ Error: datadir could not be loaded")
        return

    # Load dataset
    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    # Open file for results
    results_file = open("results_void_mIoU.txt", "a")

    # Process each batch
    for step, (images, labels, filename, _) in enumerate(loader):
        if not args.cpu:
            images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(Variable(images))

        # Special handling for BiSeNet
        if args.model == 'BiSeNet':
            new_outputs = outputs[0]
        elif args.model == 'ENet':
            new_outputs = torch.roll(outputs, -1, 1)
        else:
            new_outputs = outputs

        # Anomaly detection method selection
        if args.method == 'msp':
            softmax_probability = F.softmax(new_outputs, dim=1)
            anomaly_result = torch.argmax(softmax_probability, dim=1)
        elif args.method == 'max_logit':
            anomaly_result = torch.argmax(new_outputs, dim=1)
        elif args.method == 'max_entropy':
            softmax_probability = F.softmax(new_outputs, dim=1)
            log_softmax_probs = F.log_softmax(new_outputs, dim=1)
            entropy = -torch.sum(softmax_probability * log_softmax_probs, dim=1)
            anomaly_result = torch.argmax(entropy, dim=1)

        # Exclude void class (label 19) from evaluation
        valid_mask = labels != 19
        iouEvalVal.addBatch(anomaly_result.unsqueeze(1).data * valid_mask, labels * valid_mask)

        filenameSave = filename[0].split("leftImg8bit/")[1]
        print(step, filenameSave)
        results_file.write(f"Step {step}, File: {filenameSave}\n")

    # Compute IoU
    iouVal, iou_classes = iouEvalVal.getIoU()
    iou_classes_str = ['{:0.2f}'.format(iou_classes[i] * 100) for i in range(iou_classes.size(0))]

    # Debugging outputs
    print(f"Model: {args.model}")
    print(f"Output shape: {new_outputs.shape}")
    print(f"Min/Max values: {new_outputs.min().item()}, {new_outputs.max().item()}")
    print(f"Unique predicted classes: {torch.unique(anomaly_result)}")
    print(f"Ground truth unique classes: {torch.unique(labels)}")

    # Print and save results
    print("---------------------------------------")
    print(f"Method used: {args.method}")
    print(f"Took {time.time() - start:.2f} seconds")
    print("======================================")
    print("Per-Class IoU:")
    results_file.write(f"Method used: {args.method}\n")
    results_file.write("Per-Class IoU:\n")

    class_labels = [
        "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", "Traffic Sign",
        "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
    ]

    for i, label in enumerate(class_labels):
        print(f"{iou_classes_str[i]}% {label}")
        results_file.write(f"{label}: {iou_classes[i] * 100:.2f}%\n")

    print("======================================")
    print(f"MEAN IoU: {iouVal * 100:.2f}%")
    results_file.write(f"MEAN IoU: {iouVal * 100:.2f}%\n")
    results_file.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/content/datasets/cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'max_entropy'])
    parser.add_argument('--model', default='ERFNet', choices=['ERFNet', 'ENet', 'BiSeNet'])
    main(parser.parse_args())