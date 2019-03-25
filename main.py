import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import sys
import time
import argparse

from model import *
from loader import *

"""###################################  init  ###################################"""
parser = argparse.ArgumentParser(description='predication model')
parser.add_argument('--model', "-m", type=str, default="CNN", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--init_xavier', '-i', action='store_true', help='init with xavier')
parser.add_argument('--epoch', "-e", default=10, type=int, help='max epoch')
parser.add_argument('--predict', "-p", type=str, help='list metrics of the model')
args = parser.parse_args()


def init():
    global net
    global model_stamp

    if args.model == "CNN":
        net = CNN(embedding_dim=EmbeddingData.get_embedding_dim(),
                  num_classes=8)
    else:
        net = None
        print("no specific model")
        sys.exit(0)

    if args.resume:
        print("loading exist model from %s" % args.resume)
        check_point = torch.load(args.resume)
        net.load_state_dict(check_point["net"])
        model_stamp = args.resume[:-4]
    else:
        t = time.localtime()
        model_stamp = "%s_%d_%.2d_%.2d" % (args.model, t.tm_mday, t.tm_hour, t.tm_min)

    def xavier(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight.data)

    if args.init_xavier:
        net.apply(xavier)

    print("initializing " + model_stamp)
    net = net.cuda()


"""###################################  data loader  ###################################"""


def data_loader():
    global train_loader, val_loader

    print("loading data...")

    val_dataset = EmbeddingData("validation")

    if args.debug or args.predict:
        print("loading train dataset as the small validation dataset...")
        train_dataset = val_dataset
    else:
        train_dataset = EmbeddingData("train")

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True)


"""###################################  train  ###################################"""


def train(epoch, writer):
    global net, optimizer, criterion, train_loader
    net.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    acc = 0.0
    p = 0.0
    r = 0.0
    f1 = 0.0

    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda().float(), y.cuda().float()
            optimizer.zero_grad()

            outputs = net(x)

            loss = criterion(outputs, y)
            loss.backward()

            predicted = torch.zeros(outputs.shape).cuda()
            predicted[outputs > 0.5] = 1.0
            total_predictions += y.shape[0] * y.shape[1]
            correct_predictions += (predicted == y).sum().item()

            running_loss += loss.item()

            optimizer.step()
            torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train Loss", loss.item(), niter)

                # metrics
                p, r, f1, _ = precision_recall_fscore_support(y.cpu().numpy(), predicted.cpu().numpy())
                acc = (correct_predictions / total_predictions)
                pbar.set_postfix(loss=round(loss.item(), 4),
                                 acc_avg=round(acc, 4),
                                 )

                pbar.update(10 if pbar.n + 50 <= pbar.total else pbar.total - pbar.n)

    running_loss /= len(train_loader)

    return running_loss, acc, p, r, f1


def validate():
    global net, optimizer, criterion, val_loader
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.cuda().float(), y.cuda().float()
            outputs = net(x)

            loss = criterion(outputs, y).detach()

            predicted = torch.zeros(outputs.shape).cuda()
            predicted[outputs > 0.5] = 1.0
            total_predictions += y.shape[0] * y.shape[1]
            correct_predictions += (predicted == y).sum().item()

            running_loss += loss.item()

        running_loss /= len(val_loader)
        acc = (correct_predictions / total_predictions)
        p, r, f1, _ = precision_recall_fscore_support(y.cpu().numpy(), predicted.cpu().numpy())
        return running_loss, acc, p, r, f1


def predict():
    _, _, val_p, val_r, val_f1 = validate()
    metrics = get_metrics_df()
    for i in range(metrics.shape[0]):
        metrics.iloc[i] = val_p[i], val_r[i], val_f1[i]
    metrics = metrics.round(3)
    metrics.to_csv("result/%s_val.csv" % model_stamp)


def run_epochs():
    epoch = 0
    if args.resume:
        check_point = torch.load(args.resume)
        epoch = check_point["epoch"] + 1
    if args.predict:
        return
    elif args.debug:
        args.epoch = 1

    writer = SummaryWriter("log/%s" % model_stamp)

    if args.resume:
        train_losses = check_point["train_losses"]
        val_losses = check_point["val_losses"]
    else:
        train_losses = []
        val_losses = []

    print("start training from epoch", epoch, "-", args.epoch)
    best_val_acc = 0.
    for e in range(epoch, args.epoch):
        if args.annealing:
            scheduler.step()

        train_loss, train_acc, train_p, train_r, train_f1 = train(epoch, writer)
        val_loss, val_acc, val_p, val_r, val_f1 = validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("\re %3d: |Train avg loss: %.3f|Train avg acc: %.3f|Val avg loss: %.3f|Val avg acc: %.3f" %
              (e, train_loss, train_acc, val_loss, val_acc))

        # save check point
        if not args.debug and val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)
    writer.close()

    # store precision, recall and f1 score metrics of training and validation data
    metrics = get_metrics_df()
    for i in range(metrics.shape[0]):
        metrics.iloc[i] = train_p[i], train_r[i], train_f1[i]
    metrics = metrics.round(3)
    metrics.to_csv("result/%s_train.csv" % model_stamp)

    predict()


"""###################################  main  ###################################"""
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    init()

    data_loader()  # return train and test dataset to produce prediction

    # if isinstance(net, CenterResNet):
    #     global optimizer_centerloss, center_loss
    #     center_loss = CenterLoss(num_classes=2300, feat_dim=512) 
    #     optimizer_centerloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)

    global criterion, optimizer, scheduler
    criterion = nn.BCELoss()  # Binary Cross Entropy loss with Sigmoid
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    run_epochs()

    if args.predict:
        predict()
