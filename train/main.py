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
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile

from loss_functions import CrossEntropyLoss, FocalLoss, IsoMaxPlusLoss, LogitNormLoss
from loss_functions import IsoMaxPlus_Focal_loss, IsoMaxPlus_CE_loss, LogitNorm_Focal_loss, LogitNorm_CE_loss


NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20
# # self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
# # num_features are 16 for erfnet final conv layer
# NUM_FEATURES = 16

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, model_name, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.model_name = model_name
        self.height = height
        # pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        # need to change it for bisenet, enet, erfnet
        # if (self.enc):
        #     target = Resize(int(self.height/8), Image.NEAREST)(target)
        
        # Resize label to lower res only if the model is NOT BiSeNet and only in encoder mode
        if self.enc and self.model_name.lower() != 'bisenet':
            target = Resize(int(self.height / 8), Image.NEAREST)(target)

        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


# this calculation has been given on ENet paper***
# 19 semantic classes
# 20th class is void, used to represent anomaly/background
def compute_class_weights(dataloader, num_classes, c=1.02, ignore_index = 255): # excludes void classes
    class_count = np.zeros(num_classes)
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()
        label = label[label != ignore_index] # it excludes void classes
        flat_label = label.flatten()
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size
    
    propensity_score = class_count/total
    class_weights = 1/ (np.log(c + propensity_score + 1e-6)) # we use 1e-6 to avoid log(0) ** old version was giving zeros
    
    class_weights[19] = 0.0
    

    return torch.from_numpy(class_weights).float()


# already given encoder decoder for ERFNet
def get_class_weights(enc):
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713	
    weight[19] = 0.0
    return weight


def train(args, model, enc=False):
    
    best_acc = 0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc, args.model.lower(), augment=True, height=args.height)#1024)
    co_transform_val = MyCoTransform(enc, args.model.lower(), augment=False, height=args.height)#1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    
    if args.model.lower() == "enet":
        # ENet
        weight = compute_class_weights(loader, NUM_CLASSES)
    else:
        # ERFNet
        weight = get_class_weights(enc)
    
    
    
    try:
        if args.cuda:
            NUM_FEATURES = model.module.output_conv.in_channels
        else:
            NUM_FEATURES = model.output_conv.in_channels
    except AttributeError:
        NUM_FEATURES = 20  

    

    if args.cuda:
        weight = weight.cuda()
    
    if args.loss == "focalloss":
        criterion = FocalLoss()
    elif args.loss == "isomaxplusloss":
        criterion = IsoMaxPlusLoss(NUM_FEATURES, NUM_CLASSES)
    elif args.loss == "logitnormloss":
        criterion = LogitNormLoss()
    elif args.loss == "isomaxplusfocalloss":
        criterion = IsoMaxPlus_Focal_loss(NUM_FEATURES, NUM_CLASSES, 1/2, 1/2)
    elif args.loss == "isomaxplusceloss":
        criterion = IsoMaxPlus_CE_loss(NUM_FEATURES, NUM_CLASSES, 1/2, 1/2, weight)
    elif args.loss == "logitnormfocalloss":
        criterion = LogitNorm_Focal_loss(1/2, 1/2)
    elif args.loss == "logitnormceloss":
        criterion = LogitNorm_CE_loss(1/2, 1/2, weight)
    elif args.loss == "crossentropyloss":
        criterion = CrossEntropyLoss(weight)
        
    if args.cuda:
        criterion = criterion.cuda()
    
    
    print(type(criterion))

    savedir = args.savedir

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2

    
    start_epoch = 1
    if args.resume:
        if enc:
            filenameCheckpoint = os.path.join(savedir, 'checkpoint_enc.pth.tar')
        else:
            filenameCheckpoint = os.path.join(savedir, 'checkpoint.pth.tar')

        assert os.path.exists(filenameCheckpoint), f"Error: checkpoint not found at {filenameCheckpoint}"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        # Load model state_dict (allow partial load for flexibility)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        if missing_keys:
            print(f"[Warning] Missing model keys: {missing_keys}")
        if unexpected_keys:
            print(f"[Warning] Unexpected model keys: {unexpected_keys}")

        # Load optimizer state safely
        try:
            # Match optimizer param groups manually
            for ckpt_group, opt_group in zip(checkpoint['optimizer']['param_groups'], optimizer.param_groups):
                valid_params = [p for p in ckpt_group['params'] if p in optimizer.state]
                opt_group['params'] = valid_params

            # Copy state entries
            for key, value in checkpoint['optimizer']['state'].items():
                if key in optimizer.state:
                    optimizer.state[key] = value

            print("Optimizer state restored.")
        except Exception as e:
            print(f"[Warning] Optimizer state could not be fully restored: {e}")

        print(f"=> Resumed from checkpoint at epoch {checkpoint['epoch']}")


    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            # outputs = model(inputs, only_encode=enc)
            if args.model.lower() in ["enet", "bisenet"]:
                outputs = model(inputs)
            else:  # erfnet or others that support only_encode
                outputs = model(inputs, only_encode=enc)


            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            
            if args.model == "bisenet":
                outputs = outputs[1]
            
            loss = criterion(outputs, targets[:, 0])
            # loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                #image[0] = image[0] * .229 + .485
                #image[1] = image[1] * .224 + .456
                #image[2] = image[2] * .225 + .406
                #print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            # targets = Variable(labels, volatile=True)
            # outputs = model(inputs, only_encode=enc) 
            
            with torch.no_grad():
                inputs = images
                targets = labels
                # outputs = model(inputs, only_encode=enc)
                if args.model.lower() in ["enet", "bisenet"]:
                    outputs = model(inputs)
                else:  # erfnet or others that support only_encode
                    outputs = model(inputs, only_encode=enc)


            
            if args.model == "bisenet":
                outputs = outputs[1]

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    model_dir = "../train/"
    assert os.path.exists(model_dir + args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(model_dir + args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        
     
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """
    
    
    # PROBLEM HERE -- if erfnet encoder-decoder went well,. all good
    # if erfnet encoder is done, and the had problem, i can use decoder traning from the point of encoder.
    
    # ____________________________________________________________________________________________________

    # #train(args, model)
    # if args.model.lower() == "erfnet" and not args.decoder:
    #     print("========== ENCODER TRAINING ===========")
    #     model = train(args, model, True) #Train encoder
        
        
    #     print("========== DECODER TRAINING ===========")
    #     if (not args.state):
    #         if args.pretrainedEncoder:
    #             print("Loading encoder pretrained in imagenet")
    #             from erfnet_imagenet import ERFNet as ERFNet_imagenet
    #             pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
    #             pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
    #             pretrainedEnc = next(pretrainedEnc.children()).features.encoder
    #             if (not args.cuda):
    #                 pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
    #         else:
    #             pretrainedEnc = next(model.children()).encoder
                
                
    #         model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
            
    #         if args.cuda:
    #             model = torch.nn.DataParallel(model).cuda()
                
    #     #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    #     model = train(args, model, False)   #Train decoder
    #     print("========== TRAINING FINISHED ===========")

    # else:
    #     print("========== FULL TRAINING ==========")
    #     model = train(args, model, False)
    #     print("========== TRAINING FINISHED ==========")
    
    if args.model.lower() == "erfnet":
        if not args.decoder:
            print("========== ENCODER TRAINING ===========")
            model = train(args, model, True)

            print("========== DECODER TRAINING ===========")
            if not args.state:
                checkpoint_path = os.path.join(args.savedir, 'checkpoint_enc.pth.tar')
                assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
                checkpoint = torch.load(checkpoint_path)
                pretrainedEnc = model_file.Net(NUM_CLASSES).encoder
                pretrainedEnc.load_state_dict(checkpoint['state_dict'], strict=False)
                model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)
                if args.cuda:
                    model = torch.nn.DataParallel(model).cuda()
            model = train(args, model, False)
            print("========== TRAINING FINISHED ==========")

        else:
            print("========== DECODER TRAINING ONLY ==========")
            checkpoint_path = os.path.join(args.savedir, 'checkpoint_enc.pth.tar')
            assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
            checkpoint = torch.load(checkpoint_path)
            pretrainedEnc = model_file.Net(NUM_CLASSES).encoder
            pretrainedEnc.load_state_dict(checkpoint['state_dict'], strict=False)
            model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)
            if args.cuda:
                model = torch.nn.DataParallel(model).cuda()
            model = train(args, model, False)
            print("========== DECODER TRAINING FINISHED ==========")
    else:
        print("========== FULL TRAINING ==========")
        model = train(args, model, False)
        print("========== TRAINING FINISHED ==========")

    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default= "../datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  
    parser.add_argument('--loss', default = "crossentropyloss")

    main(parser.parse_args())