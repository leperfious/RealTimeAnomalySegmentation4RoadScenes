import os
# import cv2 # to use it, pip install opencv-python
import glob
import torch
import torch.nn.functional as F
import random

from PIL import Image
import numpy as np
from erfnet import ERFNet


import os.path as osp
from argparse import ArgumentParser


# importing metrics that we need: AuPRC and FPR95
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import roc_curve, average_precision_score

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



def main():
    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default='../datasets/validation_dataset', help='Root directory of the datasets')
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--method', type = str, default='msp', choices=['msp','max_logit', 'max_entropy'], help='Method for anomaly detection') #  check this one **
    parser.add_argument('--temperature', type=float, default=1) #  for the temperature scaling, default is 1
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="../datasets/cityscapes")  # ***
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    
    # model name, later
    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)


    # it writes to results1_ValDat.txt  ************************
    results_file = "results1_ValDat.txt"
    file = open(results_file, 'a')



    # TO ADD MORE MODEL - Start ***

    net = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(net).cuda()
    else:
        model = net


    # Update torch.load to handle warning
    if args.loadWeights.endswith('.tar'):
        model = load_my_state_dict(model, torch.load(weightspath)['state_dict'])
    else:
        model = load_my_state_dict(model, torch.load(weightspath))
    print('Model and weights LOADED successfully')
    

    # TO ADD MORE MODEL - Finish ***

    model.eval() #  starts evaluation **

    datasets = [
        "../datasets/validation_dataset/FS_LostFound_full/images/*.png",
        "../datasets/validation_dataset/fs_static/images/*.jpg",
        "../datasets/validation_dataset/RoadAnomaly/images/*.jpg",
        "../datasets/validation_dataset/RoadAnomaly21/images/*.png",
        "../datasets/validation_dataset/RoadObsticle21/images/*.webp"
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
                outputs = model(image)  # [1, C, H, W]


            # ________________________ msp, max_logit, max_entropy _______________________ starts

            
            

            if args.method == 'msp':
                softmax_probability = F.softmax(outputs/args.temperature, dim=1)  # class-wise softmax
                anomaly_score = 1.0 - torch.max(softmax_probability, dim=1)[0]  # [1, H, W]
                anomaly_result = torch.argmax(softmax_probability, dim=1).unsqueeze(1).data  # [1, 1, H, W]

            elif args.method == 'max_logit':
                anomaly_score = -torch.max(outputs, dim=1)[0]  # [1, H, W]
                anomaly_result = torch.argmax(outputs, dim=1).unsqueeze(1).data  # [1, 1, H, W]

            elif args.method == 'max_entropy':
                softmax_probability = F.softmax(outputs, dim=1)
                log_softmax_probs = F.log_softmax(outputs, dim=1)
                entropy = -torch.sum(softmax_probability * log_softmax_probs, dim=1)  # [1, H, W]
                anomaly_score = entropy
                anomaly_result = torch.argmax(softmax_probability, dim=1).unsqueeze(1).data  # [1, 1, H, W]

            # Resize anomaly_score to match GT mask resolution
            # anomaly_score will be used to calculate OOD
            
            anomaly_score = F.interpolate(
                anomaly_score.unsqueeze(1),  # [1, 1, H, W]
                size=(512, 1024),            # or ood_gts.shape if dynamic
                mode='bilinear',
                align_corners=False
            ).squeeze().data.cpu().numpy()  # → [H, W]



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
            mask = target_transform(mask)
            ood_gts = np.array(mask)

            
            # # Dataset adjustments, added here all datasets from validation
            # if "RoadAnomaly" in pathGT or "RoadAnomaly21" in pathGT:
            #     ood_gts = np.where((ood_gts==2), 1, ood_gts)

            # # I added manually, it was different
            # if "FS_LostFound_full" in pathGT:
            #     ood_gts = np.where((ood_gts==0), 0, ood_gts)
            #     ood_gts = np.where((ood_gts==1), 0, ood_gts)
            #     ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
            #     ood_gts = np.where((ood_gts == 255), 0, ood_gts)

            # # I added manually, it was not here
            # if "fs_static" in pathGT or "RoadObsticle21" in pathGT:
            #     ood_gts = np.where((ood_gts==14), 1, ood_gts)
            #     ood_gts = np.where((ood_gts<20), 0, ood_gts)
            #     ood_gts = np.where((ood_gts==255), 0, ood_gts)
            if "RoadAnomaly" in pathGT:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)
            if "LostAndFound" in pathGT:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

            if "Streethazard" in pathGT:
                ood_gts = np.where((ood_gts==14), 255, ood_gts)
                ood_gts = np.where((ood_gts<20), 0, ood_gts)
                ood_gts = np.where((ood_gts==255), 1, ood_gts)

            if 1 in np.unique(ood_gts):
                ood_gts_flat = ood_gts.flatten()
                anomaly_score_flat = anomaly_score.flatten()

                # Ensure consistent shapes
                if len(ood_gts_flat) == len(anomaly_score_flat):
                    ood_gts_list.append(ood_gts_flat)
                    anomaly_score_list.append(anomaly_score_flat)
                else:
                    print(f"Inconsistent shapes: ood_gts_flat={len(ood_gts_flat)}, anomaly_score_flat={len(anomaly_score_flat)}")

            
            del outputs, anomaly_score, image
            torch.cuda.empty_cache()



        # # Check if any data was processed
        # if len(ood_gts_list) == 0 or len(anomaly_score_list) == 0:
        #     print("No valid data processed. Ensure dataset is correct.")
        #     return


        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)

        ood_mask = ood_gts == 1
        ind_mask = ood_gts == 0

        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        
        # if len(np.unique(val_label)) < 2:
        #     prc_auc = np.nan
        #     fpr = np.nan
        # else:
        #     prc_auc = average_precision_score(val_label, val_out)
        #     fpr = fpr_at_95_tpr(val_out, val_label)


        dataset_name = dataset_path.split("/")[-3]
        # print(f'Model: {modelname.upper()}')
        print(f'Method: {args.method}')
        print(f'Temperature: {args.temperature}')
        print(f'Dataset: {dataset_name}')
        print(f'AUPRC score: {round(prc_auc*100.0, 3)}')
        print(f'FPR@TPR95: {round(fpr*100.0, 3)}')

        file.write(
            f'Method: {args.method}      Temperature: {args.temperature}       Dataset: {dataset_name}   AUPRC score: {round(prc_auc * 100.0, 3)}   FPR@TPR95: {round(fpr * 100.0, 3)}\n'
        )

    file.close()

if __name__ == '__main__':
    main()