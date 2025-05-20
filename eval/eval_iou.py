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
# from erfnet import ERFNet

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

def load_my_state_dict(model, state_dict, loadModel):  #custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param)
        elif name.startswith("module.") and name[7:] in own_state: # [7:] to delete module.encoder.weight to encoder.weight -- hardcoded :p
            own_state[name[7:]].copy_param()
        elif loadModel != "erfnet.py" and ("module." + name) in own_state:
            own_state["module." + name].copy_param()
        else:
            print(f"{name} not loaded.")
    return model

def main(args):
    
    model_module = importlib.import_module(args.loadModel[:-3]) # name are not Enet or Erfnet, they are Net
    model = model_module.Net(NUM_CLASSES)


    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()
    
    
    model = load_my_state_dict(model, torch.load(weightspath, map_location='cpu', weights_only=True), args.loadModel)
    
    model.eval() # evaluation starts


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, 
                                   subset=args.subset),
                         num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    # if we do it with the void, it is different
    if args.void:
        iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=20) # include void, 20 in total
    else:
        iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=19) # ignore void

    start = time.time()

    # To write results tab1 to file results1_mIoU.txt
    results_file = open("results_mIoU.txt", "a")

    for step, (images, labels, filename, _ ) in enumerate(loader):
        if not args.cpu:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            outputs = model(images)
        
        if args.loadModel == "bisenet.py":
            outputs = outputs[1]

        # ________________________ msp, max_logit, max_entropy _______________________ starts
        # size is 512x1024

        if args.method == 'msp':
            softmax_probability = F.softmax(outputs/args.temperature, dim=1)
            anomaly_score = 1.0 - torch.max(softmax_probability, dim=1)[0]
            anomaly_result = torch.argmax(softmax_probability, dim=1).unsqueeze(1).data  # use for mIoU

        elif args.method == 'max_logit':
            anomaly_score = -torch.max(outputs, dim=1)[0]
            anomaly_result = torch.argmax(outputs, dim=1).unsqueeze(1).data  # use for mIoU

        elif args.method == 'max_entropy':
            softmax_probability = F.softmax(outputs, dim=1)
            log_softmax_probs = F.log_softmax(outputs, dim=1)
            entropy = -torch.sum(softmax_probability * log_softmax_probs, dim=1)
            anomaly_score = entropy
            anomaly_result = torch.argmax(softmax_probability, dim=1).unsqueeze(1).data  # use for mIoU

        # anomaly_score will be used to calculate OOD



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
    print("Temperature scale used:", args.temperature)
    print("Took", time.time() - start, "seconds")
    print("======================================")
    print("Per-Class IoU:")
    results_file.write(f"Method used: {args.method}\n")
    results_file.write(f"Temperature scale used: {args.temperature}\n")
    results_file.write("Per-Class IoU:\n")
    class_labels=["Road", 
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
                "Bicycle"
                ]
    if args.void:
        class_labels.append("Void")
    
    for i, label in enumerate(class_labels):
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
    parser.add_argument('--temperature', type=float, default=1) #  for the temperature scaling, default is 1
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="../datasets/cityscapes")  # It needed to be corrected, i do locally here
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'max_entropy'], help='Method for anomaly detection')
    parser.add_argument('--void', action='store_true') # this one works for 19 and 19+1
    main(parser.parse_args())
    
    # when running python adding --void with include it, without --void, it will exclude