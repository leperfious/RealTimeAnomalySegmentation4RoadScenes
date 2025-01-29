# Main code for training ENet model in Cityscapes dataset
# Updated: Jan 2025
# Converted from ERFNet to ENet with user-defined weights

import os
import random
import time
import numpy as np
import torch

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

NUM_CLASSES = 20  # Cityscapes classes

color_transform = Colorize(NUM_CLASSES)

# ✅ Precomputed Class Weights for ENet
ENet_weights = torch.tensor([
    4.363572, 7.188593, 5.087264, 9.437396, 9.276026, 9.045236, 9.804637, 9.516575, 5.662319, 9.088848,
    7.784346, 9.05168, 9.871636, 6.975896, 9.752323, 9.780681, 9.782797, 9.905354, 9.628174, 10.0
])

# ✅ Data Augmentations
class MyCoTransform(object):
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize((self.height, self.height), Image.BILINEAR)(input)
        target = Resize((self.height, self.height), Image.NEAREST)(target)

        if self.augment:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)
            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target

# ✅ Loss Function
class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

# ✅ Training Function
def train(args, model):
    best_acc = 0

    assert os.path.exists(args.datadir), "Error: Dataset directory not found!"

    # ✅ Dataset Preparation
    co_transform = MyCoTransform(augment=True, height=args.height)
    co_transform_val = MyCoTransform(augment=False, height=args.height)

    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    if len(dataset_train) == 0 or len(dataset_val) == 0:
        raise RuntimeError("❌ Dataset is empty! Check dataset path.")

    # ✅ Assign Precomputed Weights
    weight = ENet_weights.cuda() if args.cuda else ENet_weights
    criterion = CrossEntropyLoss2d(weight)

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    savedir = f'../save/{args.savedir}'
    os.makedirs(savedir, exist_ok=True)

    log_path = os.path.join(savedir, "automated_log.txt")
    model_txt_path = os.path.join(savedir, "model.txt")

    if not os.path.exists(log_path):
        with open(log_path, "a") as myfile:
            myfile.write("Epoch\tTrain-loss\tVal-loss\tTrain-IoU\tVal-IoU\tLR")

    with open(model_txt_path, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    start_epoch = 1

    if args.resume:
        checkpoint_path = os.path.join(savedir, 'checkpoint.pth.tar')
        assert os.path.exists(checkpoint_path), "Error: No checkpoint found!"
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print(f"=> Loaded checkpoint at epoch {checkpoint['epoch']}")

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"----- TRAINING - EPOCH {epoch} -----")
        model.train()

        epoch_loss = []
        scheduler.step(epoch)

        iouEvalTrain = iouEval(NUM_CLASSES) if args.iouTrain else None

        for images, labels in loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            inputs, targets = Variable(images), Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

            if args.iouTrain:
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

        avg_loss_train = sum(epoch_loss) / len(epoch_loss)
        iouTrain = iouEvalTrain.getIoU()[0] if args.iouTrain else 0

        print(f"EPOCH IoU on TRAIN set: {iouTrain:.4f}%")

        # ✅ Validation Step
        model.eval()
        epoch_loss_val = []
        iouEvalVal = iouEval(NUM_CLASSES) if args.iouVal else None

        with torch.no_grad():
            for images, labels in loader_val:
                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                inputs, targets = Variable(images), Variable(labels)
                outputs = model(inputs)

                loss = criterion(outputs, targets[:, 0])
                epoch_loss_val.append(loss.item())

                if args.iouVal:
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

        avg_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        iouVal = iouEvalVal.getIoU()[0] if args.iouVal else 0

        print(f"EPOCH IoU on VAL set: {iouVal:.4f}%")

        # ✅ Save Model Checkpoints
        is_best = iouVal > best_acc
        best_acc = max(iouVal, best_acc)

        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()},
                   f"{savedir}/checkpoint.pth.tar")

        if is_best:
            torch.save(model.state_dict(), f"{savedir}/model_best.pth")
            print(f"Best model saved: {savedir}/model_best.pth (Epoch: {epoch})")

        # ✅ Log Training Stats
        with open(log_path, "a") as myfile:
            myfile.write(f"\n{epoch}\t{avg_loss_train:.4f}\t{avg_loss_val:.4f}\t{iouTrain:.4f}\t{iouVal:.4f}\t{scheduler.get_last_lr()[0]:.8f}")

    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="enet")
    parser.add_argument('--datadir', default=os.getenv("CITYSCAPES_DATASET", "/content/datasets/cityscapes/"))  
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)

    main(parser.parse_args())