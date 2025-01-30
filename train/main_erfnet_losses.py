import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

import importlib
from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20

# Augmentations
class MyCoTransform:
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize((self.height, self.height), Image.BILINEAR)(input)
        target = Resize((self.height, self.height), Image.NEAREST)(target)
        
        if self.augment:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        
        input = ToTensor()(input)
        if self.enc:
            target = Resize((self.height // 8, self.height // 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target

# Loss functions
class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
        prob = torch.exp(log_prob)
        return self.loss((1 - prob) ** self.gamma * log_prob, targets)

# Training function
def train(args, model, enc=False):
    best_acc = 0

    weight = torch.ones(NUM_CLASSES)
    weight[19] = 0
    if enc:
        weight[0:19] = torch.tensor([2.365, 4.423, 2.969, 5.344, 5.298, 5.227, 5.439, 5.365, 3.417, 
                                      5.241, 4.737, 5.228, 5.455, 4.301, 5.426, 5.433, 5.433, 5.463, 5.394])
    else:
        weight[0:19] = torch.tensor([2.814, 6.985, 3.789, 9.942, 9.770, 9.511, 10.311, 10.026, 4.632,
                                      9.560, 7.869, 9.516, 10.373, 6.661, 10.260, 10.287, 10.289, 10.405, 10.138])
    
    weight = weight.cuda() if args.cuda else weight
    
    if args.loss2 == "cross_entropy":
        criterion = CrossEntropyLoss2d(weight)
    elif args.loss2 == "focal_loss":
        criterion = FocalLoss(weight=weight)
    else:
        raise ValueError("Unsupported loss function")
    
    co_transform = MyCoTransform(enc, augment=True, height=args.height)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9))
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        scheduler.step(epoch)
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images, only_encode=enc)
            loss = criterion(outputs, labels[:, 0])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
        print(f"Epoch {epoch}: Loss {sum(epoch_loss) / len(epoch_loss):.4f}")

        # Validation
        model.eval()
        val_loss = []
        iouEvalVal = iouEval(NUM_CLASSES)
        with torch.no_grad():
            for images, labels in loader_val:
                if args.cuda:
                    images, labels = images.cuda(), labels.cuda()
                outputs = model(images, only_encode=enc)
                loss = criterion(outputs, labels[:, 0])
                val_loss.append(loss.item())
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1), labels)
        
        iouVal, _ = iouEvalVal.getIoU()
        print(f"Validation IoU: {iouVal:.4f}")
        
        if iouVal > best_acc:
            best_acc = iouVal
            torch.save(model.state_dict(), f"../save/{args.savedir}/best_model.pth")
    
    return model

# Main function
def main(args):  
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        model.load_state_dict(torch.load(args.state))


    if not args.decoder:
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  

    
    print("========== DECODER TRAINING ===========")
    if not args.state:
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in ImageNet...")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if not args.cuda:
                pretrainedEnc = pretrainedEnc.cpu()  # Move to CPU if needed
        else:
            pretrainedEnc = next(model.children()).encoder

        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  # Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()

    model = train(args, model, False)  # ✅ Pass args to train()
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help="Enable CUDA if available")
    parser.add_argument('--model', default="erfnet", help="Model to train")
    parser.add_argument('--state', help="Path to checkpoint for resuming training")

    parser.add_argument('--port', type=int, default=8097, help="Port for visualization dashboard")
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/", help="Path to dataset")
    parser.add_argument('--height', type=int, default=512, help="Input image height")
    parser.add_argument('--num-epochs', type=int, default=20, help="Total number of epochs")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of data loading workers")
    parser.add_argument('--batch-size', type=int, default=6, help="Batch size for training")
    parser.add_argument('--steps-loss', type=int, default=50, help="Steps interval for loss logging")
    parser.add_argument('--steps-plot', type=int, default=50, help="Steps interval for visualization")
    parser.add_argument('--epochs-save', type=int, default=0, help="Save model every X epochs")
    parser.add_argument('--savedir', required=True, help="Directory to save model checkpoints and logs")

    parser.add_argument('--decoder', action='store_true', help="Train only the decoder")
    parser.add_argument('--pretrainedEncoder', help="Path to ImageNet-pretrained ERFNet encoder")
    parser.add_argument('--visualize', action='store_true', help="Enable visualization during training")

    parser.add_argument('--iouTrain', action='store_true', default=False, help="Compute IoU during training (slower)")
    parser.add_argument('--iouVal', action='store_true', default=True, help="Compute IoU during validation")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")

    parser.add_argument('--loss1', type=str, default="logit_norm", choices=['logit_norm', 'isomax'],
                        help="First loss function (logit_norm / isomax)")
    parser.add_argument('--loss2', type=str, default="cross_entropy", choices=['cross_entropy', 'focal_loss'],
                        help="Second loss function (cross_entropy / focal_loss)")

    args = parser.parse_args()
    main(args)