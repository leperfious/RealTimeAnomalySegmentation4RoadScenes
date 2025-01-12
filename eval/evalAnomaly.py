import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet


import os.path as osp
from argparse import ArgumentParser


# importing metrics that we need: AuPRC and FPR95
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

# input_transform
from torchvision.transforms import Resize, Compose, ToTensor

seed = 42

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# OK - no extra
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

# FPR95 implementation **
def fpr_at_95_tpr(y_scores, y_true):
    fpr, tpr, thresholds = roc_curve (y_true, y_scores)
    close_tpr = np.abs(tpr-0.95).argmin()
    return fpr[close_tpr]


def main():
    parser = ArgumentParser()

    """""
    parser.add_argument(
        "--input",
        default="/content/validation_dataset/RoadObsticle21/images/*.webp", #  check this one ** 
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    """

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--method', default='msp', choices=['msp','max_logit', 'max_entropy'], help='Method for anomaly detection') #  check this one **
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    # we use datadir for training the model, but we use testdir on pre-trained models
    # parser.add_argument('--datadir', default="/content/datasets/cityscapes")  # ***
    parser.add_argument('--testdir', default="/content/datasets/validation_dataset")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    
    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


    # it writes to results1_ValDat.txt  ************************
    results_file = "results1_ValDat.txt"
    if not os.path.exists(results_file):
        open(results_file, 'w').close()



    # TO ADD MORE MODEL - Start ***

    model = ERFNet(NUM_CLASSES)


    # Update torch.load to handle warning
    try:
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True))
    except TypeError:  # For older PyTorch versions
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))


    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()


        
    # TO ADD MORE MODEL - Finish ***

    model.eval() #  starts evaluation **

    datasets = [
        "/content/datasets/validation_dataset/FS_LostFound_full/images/*.png",
        "/content/datasets/validation_dataset/fs_static/images/*.jpg",
        "/content/datasets/validation_dataset/RoadAnomaly/images/*.jpg",
        "/content/datasets/validation_dataset/RoadAnomaly21/images/*.png",
        "/content/datasets/validation_dataset/RoadObsticle21/images/*.webp"
    ]

    for dataset_path in datasets:
        print(f"Processing dataset: {dataset_path}")

        anomaly_score_list = []
        ood_gts_list = []

        for path in glob.glob(os.path.expanduser(dataset_path)):
            print("Processing file:", path)

            # images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
            # images = images.permute(0,3,1,2) 
            # it manually changes to pytorch format, but we can use instead of out_transform

            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float() #  I don't use ToTensor() because we will do resize, data augmentation etc.
            if not args.cpu:
                image = image.cuda()

            with torch.no_grad():
                outputs = model(image)

            # ________________________ msp, max_logit, max_entropy _______________________ starts

            if args.method == 'msp':
                #  temperature = float(args.temperature) # when we use temperatue, we need to also divide ouputs to it
                softmax_probability = torch.nn.functional.softmax(outputs, dim = 1) # dim 0 operates across the batch dimension, dim 1 applies softmax operation across num_classes dimentions for each pixel
                anomaly_score = 1.0 - torch.max(softmax_probability, dim=1)[0]
            elif args.method == 'max_logit':
                anomaly_score = -torch.max(outputs, dim=1)[0]
            elif args.method == 'max_entropy':
                softmax_probability = torch.nn.functional.softmax(outputs, dim = 1)
                log_softmax_probs = torch.nn.functional.log_softmax(outputs, dim = 1)
                entropy = -torch.sum(softmax_probability * log_softmax_probs, dim = 1)
                anomaly_score = entropy

            # ________________________ msp, max_logit, max_entropy _______________________ ends

            # Testing datasets are different from each other, we need to fix it.

            pathGT = path.replace("images", "labels_masks") #  instead of changing them by ourselves, we change it to labels_masks format. We can make it easier
            for ext in ["png", "jpg", "webp"]:
                if os.path.exists(pathGT.replace(path.split('.')[-1], ext)):
                    pathGT = pathGT.replace(path.split('.')[-1], ext)
                    break
            else:
                print(f"Ground truth is not here: {path}")
                continue


            mask = Image.open(pathGT)
            ood_gts = np.array(mask)

            
            # Dataset adjustments, added here all datasets from validation
            if "RoadAnomaly" in pathGT or "RoadAnomaly21" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)

            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

            if "fs_static" in pathGT or "RoadObsticle21" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 in np.unique(ood_gts):
                ood_gts_list.append(ood_gts)
                if isinstance(anomaly_score, torch.Tensor):
                    anomaly_score = anomaly_score.detach().cpu().numpy()
                anomaly_score_list.append(anomaly_score)



        print(f"Anomaly score list length: {len(anomaly_score_list)}")
        print(f"OOD ground truth list length: {len(ood_gts_list)}")

        val_label = np.concatenate([mask.flatten() for mask in ood_gts_list])
        val_out = np.concatenate([out.flatten() for out in anomaly_score_list])

        print(f"val_out length: {len(val_out)}")
        print(f"val_label length: {len(val_label)}")

        au_prc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)

        print(f"AuPRC for {dataset_path}: {au_prc * 100.0}%")
        print(f"FPR95 for {dataset_path}: {fpr * 100.0}%")

        with open(results_file, 'a') as file:
            file.write(f"{dataset_path} - AuPRC: {au_prc * 100.0:.2f}%, FPR@95: {fpr * 100.0:.2f}%\n")


if __name__ == '__main__':
    main()