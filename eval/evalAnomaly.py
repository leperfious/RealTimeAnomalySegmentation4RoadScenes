import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor
from argparse import ArgumentParser
from erfnet import ERFNet

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Constants
NUM_CLASSES = 20

# Input transformations
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

def load_my_state_dict(model, state_dict):
    """Custom function to load model state dict."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            own_state[name].copy_(param)
        else:
            print(f"{name} not loaded")
    return model

def fpr_at_95_tpr(y_scores, y_true):
    """Calculate FPR at 95% TPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    close_tpr = np.abs(tpr - 0.95).argmin()
    return fpr[close_tpr]

def main():
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--method', default='msp', choices=['msp', 'max_logit', 'max_entropy'], help='Anomaly detection method')
    parser.add_argument('--testdir', default="/content/datasets/validation_dataset", help='Directory containing test datasets')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Load model
    model = ERFNet(NUM_CLASSES)
    weightspath = os.path.join(args.loadDir, args.loadWeights)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()
    model = load_my_state_dict(model, torch.load(weightspath, map_location='cpu'))
    model.eval()

    datasets = [
        os.path.join(args.testdir, "FS_LostFound_full/images/*.png"),
        os.path.join(args.testdir, "fs_static/images/*.jpg"),
        os.path.join(args.testdir, "RoadAnomaly/images/*.jpg"),
        os.path.join(args.testdir, "RoadAnomaly21/images/*.png"),
        os.path.join(args.testdir, "RoadObsticle21/images/*.webp")
    ]

    results_file = "results1_ValDat.txt"
    if not os.path.exists(results_file):
        open(results_file, 'w').close()

    with open(results_file, 'a') as file:
        for dataset_path in datasets:
            print(f"Processing dataset: {dataset_path}")
            anomaly_score_list = []
            ood_gts_list = []

            for path in glob.glob(dataset_path):
                print(f"Processing file: {path}")

                # Load and preprocess image
                image = input_transform(Image.open(path).convert('RGB')).unsqueeze(0)
                if not args.cpu:
                    image = image.cuda()

                with torch.no_grad():
                    outputs = model(image).cpu()

                if args.method == 'msp':
                    softmax_prob = torch.nn.functional.softmax(outputs, dim=1)
                    anomaly_score = 1.0 - torch.max(softmax_prob, dim=1)[0].numpy()
                elif args.method == 'max_logit':
                    anomaly_score = -torch.max(outputs, dim=1)[0].numpy()
                elif args.method == 'max_entropy':
                    softmax_prob = torch.nn.functional.softmax(outputs, dim=1)
                    log_softmax = torch.nn.functional.log_softmax(outputs, dim=1)
                    entropy = -torch.sum(softmax_prob * log_softmax, dim=1).numpy()
                    anomaly_score = entropy

                # Process ground truth
                pathGT = path.replace("images", "labels_masks")
                pathGT = pathGT.rsplit('.', 1)[0] + ".png"

                if not os.path.exists(pathGT):
                    print(f"Ground truth not found for: {path}")
                    continue

                mask = Image.open(pathGT)
                ood_gts = np.array(mask)

                if "RoadAnomaly" in pathGT or "RoadAnomaly21" in pathGT:
                    ood_gts = np.where(ood_gts == 2, 1, ood_gts)

                if 1 in np.unique(ood_gts):
                    ood_gts_list.append(ood_gts.flatten())
                    anomaly_score_list.append(anomaly_score.flatten())

            # Calculate metrics
            val_label = np.concatenate(ood_gts_list)
            val_out = np.concatenate(anomaly_score_list)

            if len(val_label) != len(val_out):
                print("Error: Mismatch in lengths between val_label and val_out")
                continue

            au_prc = average_precision_score(val_label, val_out)
            fpr = fpr_at_95_tpr(val_out, val_label)

            print(f"Dataset: {dataset_path}")
            print(f"AUPRC: {au_prc * 100:.2f}%")
            print(f"FPR@95: {fpr * 100:.2f}%")

            file.write(f"{dataset_path} - AUPRC: {au_prc * 100:.2f}%, FPR@95: {fpr * 100:.2f}%\n")

if __name__ == '__main__':
    main()
