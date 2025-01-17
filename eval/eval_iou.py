# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F  # msp, maxlogit, maxentropy 
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

def main(args):

    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    # model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    # to add

    # if not os.path.exists(args.datadir):
    #     raise FileNotFoundError(f"Dataset directory not found: {args.datadir}")
    # print(f"Using dataset directory: {args.datadir}")

    # Update torch.load to handle warning
    try:
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True))
    except TypeError:  # For older PyTorch versions
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))


    # model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    # print ("Model and weights LOADED successfully")

    model.eval() # evaluation starts

    # if(not os.path.exists(args.datadir)):
    #     print ("Error: datadir could not be loaded")
    #     return


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
                         num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    # To write results tab1 to file results1_mIoU.txt
    results_file = open("results1_mIoU.txt", "a")

    for step, (images, labels, filename, _ ) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            outputs = model(Variable(images))

        # ________________________ msp, max_logit, max_entropy _______________________ starts

        if args.method == 'msp':
            softmax_probability = F.softmax(outputs, dim=0)  # Changed from dim=1 to dim=0
            anomaly_result = torch.argmax(softmax_probability, dim=0).unsqueeze(1).data
        elif args.method == 'max_logit':
            anomaly_result = torch.argmax(outputs, dim=0).unsqueeze(1).data  # Changed from dim=1 to dim=0
        elif args.method == 'max_entropy':
            softmax_probability = F.softmax(outputs, dim=0)  # Changed from dim=1 to dim=0
            log_softmax_probs = F.log_softmax(outputs, dim=0)  # Changed from dim=1 to dim=0
            entropy = -torch.sum(softmax_probability * log_softmax_probs, dim=0)  # Changed from dim=1 to dim=0
            anomaly_result = torch.argmax(entropy, dim=0).unsqueeze(1).data  # Changed from dim=1 to dim=0

        # ________________________ msp, max_logit, max_entropy _______________________ ends

        iouEvalVal.addBatch(anomaly_result, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 
        print(step, filenameSave)
        results_file.write(f"Step {step}, File: {filenameSave}\n")


    iouVal, iou_classes = iouEvalVal.getIoU()
    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Method used:", args.method)
    print("Took", time.time() - start, "seconds")
    print("======================================")
    print("Per-Class IoU:")
    results_file.write(f"Method used: {args.method}\n")
    results_file.write("Per-Class IoU:\n")
    for i, label in enumerate(["Road", 
                               "Sidewalk", 
                               "Building", 
                               "Wall", 
                               "Fence", 
                               "Pole", 
                               "Traffic Light", 
                               "Traffic Sign", 
                               "Vegetation", 
                               "Terrain", 
                               "Sky", 
                               "Person", 
                               "Rider", 
                               "Car", 
                               "Truck", 
                               "Bus", 
                               "Train", 
                               "Motorcycle", 
                               "Bicycle"]):
        print(iou_classes_str[i], label)
        results_file.write(f"{label}: {iou_classes[i]*100:.2f}%\n")
    print("======================================")
    print("MEAN IoU:", f'{iouVal*100:.2f}', "%")
    results_file.write(f"MEAN IoU: {iouVal*100:.2f}%\n")

    results_file.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/content/datasets/cityscapes")  # It needed to be corrected, i bring it from colab acc.
    parser.add_argument('--testdir', default="/content/datasets/validation_dataset")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'max_entropy'], help='Method for anomaly detection')
    main(parser.parse_args())