import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import importlib
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image  

from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval

# Number of classes in Cityscapes
NUM_CLASSES = 20


CLASS_WEIGHTS = torch.tensor([
    2.8149, 6.9850, 3.7890, 9.9428, 9.7702,
    9.5111, 10.3113, 10.0265, 4.6323, 9.5608,
    7.8698, 9.5169, 10.3737, 6.6616, 10.2605,
    10.2879, 10.2898, 10.4053, 10.1381, 0
], dtype=torch.float32).cuda()

class LogitNormLoss(nn.Module):
    def forward(self, outputs, targets):
        normed_logits = outputs / torch.norm(outputs, p=2, dim=1, keepdim=True)
        return F.cross_entropy(normed_logits, targets, weight=CLASS_WEIGHTS)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    
    def forward(self, outputs, targets):
        ce_loss = self.ce(outputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

class IsotropicMaxLoss(nn.Module):
    def forward(self, outputs, targets):
        iso_loss = torch.mean(torch.sum(outputs ** 2, dim=1))
        ce_loss = F.cross_entropy(outputs, targets, weight=CLASS_WEIGHTS)
        return ce_loss + iso_loss

def get_loss_function(loss_type):
    if loss_type == "LN+CE":
        return LogitNormLoss()
    elif loss_type == "LN+FL":
        return nn.ModuleList([LogitNormLoss(), FocalLoss()])
    elif loss_type == "Isomax+CE":
        return IsotropicMaxLoss()
    elif loss_type == "Isomax+FL":
        return nn.ModuleList([IsotropicMaxLoss(), FocalLoss()])
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


class MyCoTransform:
    def __init__(self, height=512):
        self.height = height
    
    def __call__(self, input, target):
        input = Resize((self.height, self.height), interpolation=Image.BILINEAR)(input)
        target = Resize((self.height, self.height), interpolation=Image.NEAREST)(target)
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target

def train(args, model, loss_fn, loss_name):
    best_acc = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    co_transform = MyCoTransform(height=args.height)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform, 'val')

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - (epoch / args.num_epochs)) ** 0.9)

    save_dir = f'save/erfnet_{loss_name}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs} - Training with {loss_name}")
        model.train()
        epoch_loss = 0.0
        iou_eval = iouEval(NUM_CLASSES)

        for images, targets in loader_train:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(loss_fn, nn.ModuleList):  # Handle combined losses
                loss = sum(l(outputs, targets[:, 0]) for l in loss_fn)
            else:
                loss = loss_fn(outputs, targets[:, 0])

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            iou_eval.addBatch(outputs.max(1)[1].unsqueeze(1), targets.unsqueeze(1))

        avg_loss = epoch_loss / len(loader_train)
        train_iou = iou_eval.getIoU()[0] * 100
        print(f"Train Loss: {avg_loss:.4f} | Train IoU: {train_iou:.2f}%")

        # ✅ Validation
        model.eval()
        val_loss = 0.0
        iou_eval_val = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for images, targets in loader_val:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)

                if isinstance(loss_fn, nn.ModuleList):
                    loss = sum(l(outputs, targets[:, 0]) for l in loss_fn)
                else:
                    loss = loss_fn(outputs, targets[:, 0])

                val_loss += loss.item()
                iou_eval_val.addBatch(outputs.max(1)[1].unsqueeze(1), targets.unsqueeze(1))

        avg_val_loss = val_loss / len(loader_val)
        val_iou = iou_eval_val.getIoU()[0] * 100
        print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.2f}%")

        scheduler.step()
        if val_iou > best_acc:
            best_acc = val_iou
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"New best model saved at {save_dir}/best_model.pth")

        torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch}.pth")


def load_pretrained_model(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

def main(args):
    print("Loading Pretrained ERFNet Model...")
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    load_pretrained_model(model, args.state)

    for loss_name in ["LN+CE", "LN+FL", "Isomax+CE", "Isomax+FL"]:
        loss_fn = get_loss_function(loss_name)
        train(args, model, loss_fn, loss_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--datadir', default="/content/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--savedir', type=str, required=True, help="Directory to save the model checkpoints")
    parser.add_argument('--state', type=str, required=True, help="Path to the pretrained model")
    parser.add_argument('--loss1', type=str, required=True, choices=['logit_norm', 'isomax'])
    parser.add_argument('--loss2', type=str, required=True, choices=['cross_entropy', 'focal_loss'])

    args = parser.parse_args()
    main(args)
