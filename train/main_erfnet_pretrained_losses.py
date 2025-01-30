import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import importlib
from argparse import ArgumentParser
from dataset import cityscapes
from transform import Relabel, ToLabel
from torchvision.transforms import Compose, Resize, ToTensor
from iouEval import iouEval
from PIL import Image

# Number of classes in Cityscapes
NUM_CLASSES = 20

# Define Loss Functions
class LogitNormLoss(nn.Module):
    """Logit Normalization Loss"""
    def __init__(self):
        super(LogitNormLoss, self).__init__()
    
    def forward(self, outputs, targets):
        normed_logits = outputs / torch.norm(outputs, p=2, dim=1, keepdim=True)
        return F.cross_entropy(normed_logits, targets)

class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, outputs, targets):
        ce_loss = self.ce(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class IsotropicMaxLoss(nn.Module):
    """Isotropic Maximization Loss"""
    def __init__(self):
        super(IsotropicMaxLoss, self).__init__()
    
    def forward(self, outputs, targets):
        iso_loss = torch.mean(torch.sum(outputs ** 2, dim=1))
        ce_loss = F.cross_entropy(outputs, targets)
        return ce_loss + iso_loss

def get_loss_function(loss1, loss2):
    """Return the correct loss function based on user arguments"""
    loss_functions = {
        "logit_norm": LogitNormLoss(),
        "isomax": IsotropicMaxLoss(),
        "cross_entropy": nn.CrossEntropyLoss(),
        "focal_loss": FocalLoss(),
    }

    if loss1 not in loss_functions or loss2 not in loss_functions:
        raise ValueError(f"Invalid loss function names: {loss1}, {loss2}")

    def combined_loss(outputs, targets):
        return loss_functions[loss1](outputs, targets) + loss_functions[loss2](outputs, targets)

    return combined_loss

# Data Transformations
class MyCoTransform:
    def __init__(self, height=512):
        self.height = height
    
    def __call__(self, input, target):
        input = Resize((self.height, self.height), interpolation=Image.BILINEAR)(input)
        target = Resize((self.height, self.height), interpolation=Image.NEAREST)(target)  # FIX: NEAREST interpolation
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)  # FIX: Ensure label mapping is correct
        return input, target

def train(args, model, loss_fn, loss_name):
    """Training Loop"""
    best_acc = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load dataset
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
        class_weights = torch.tensor([
            2.8149201869965, 6.9850029945374, 3.7890393733978, 9.9428062438965,
            9.7702074050903, 9.5110931396484, 10.311357498169, 10.026463508606,
            4.6323022842407, 9.5608062744141, 7.8698215484619, 9.5168733596802,
            10.373730659485, 6.6616044044495, 10.260489463806, 10.287888526917,
            10.289801597595, 10.405355453491, 10.138095855713, 0  # 19th class ignored
            ])

        iou_eval = iouEval(NUM_CLASSES)

        for images, targets in loader_train:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            iou_eval.addBatch(outputs.max(1)[1], targets)

        avg_loss = epoch_loss / len(loader_train)
        train_iou = iou_eval.getIoU()[0] * 100
        print(f"Train Loss: {avg_loss:.4f} | Train IoU: {train_iou:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        iou_eval_val = iouEval(NUM_CLASSES)

        with torch.no_grad():
            for images, targets in loader_val:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets[:, 0])
                val_loss += loss.item()
                iou_eval_val.addBatch(outputs.max(1)[1], targets)

        avg_val_loss = val_loss / len(loader_val)
        val_iou = iou_eval_val.getIoU()[0] * 100
        print(f"Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.2f}%")

        scheduler.step()

        # Save best model
        if val_iou > best_acc:
            best_acc = val_iou
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"New best model saved at {save_dir}/best_model.pth")

        # Save every epoch
        torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch}.pth")

def load_pretrained_model(model, pretrained_path):
    """Load pretrained weights into the model"""
    print(f"Loading Pretrained Model from: {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Ensure we correctly handle cases where the checkpoint has or doesn't have 'state_dict'
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove "module." prefix if necessary
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=False)

    print("Pretrained model loaded successfully!")
    return model

def main(args):
    """Main Training Function"""
    print("Loading Pretrained ERFNet Model...")

    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)

    if not os.path.exists(args.state):
        raise FileNotFoundError(f"Pretrained model file not found: {args.state}")

    model = load_pretrained_model(model, args.state)

    loss_name = f"{args.loss1}_{args.loss2}"
    loss_fn = get_loss_function(args.loss1, args.loss2)
    
    print(f"\n========== Training with {loss_name} ==========")
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