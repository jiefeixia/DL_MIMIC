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
from sklearn.metrics import f1_score

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
parser.add_argument('--model', "-m", type=str, default="LSTM", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--init_xavier', '-i', action='store_true', help='init with xavier')
parser.add_argument('--epoch', "-e", default=20, type=int, help='max epoch')
parser.add_argument('--predict', "-p", type=str, help='list metrics of the model')
args = parser.parse_args()


def init():
    global net
    global model_stamp

    if args.model == "CNN":
        net = CNN(vocab_size=IdxData.get_vacab_size(),
                  embedding_dim=EmbeddingData.get_embedding_dim(),
                  num_classes=8)
    elif args.model == "LSTM":
        net = LSTM(vocab_size=IdxData.get_vacab_size(),
                   embedding_dim=EmbeddingData.get_embedding_dim(),
                   hidden_size=512,
                   layers=3,
                   dropout=0,
                   num_classes=8)
    else:
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
    global train_loader, val_loader, train_dataset, val_dataset, test_dataset, test_loader

    print("loading data...")
    test_dataset = Data("test")
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=8,
                             shuffle=False,
                             collate_fn=collate)

    # val_dataset = EmbeddingData("validation")
    val_dataset = Data("validation")

    if args.debug or args.predict:
        print("loading train dataset as the small validation dataset...")
        train_dataset = val_dataset
    else:
        # train_dataset = EmbeddingData("train")
        train_dataset = Data("train")

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True,
                            collate_fn=collate)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              collate_fn=collate)


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
        for batch_idx, (x, seq_len, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            outputs = net(x, seq_len)

            outputs = outputs.cuda().double()
            y = y.cuda().double()
            loss = criterion(outputs, y)
            loss.backward()

            pred = torch.zeros(outputs.shape).cuda()
            pred[outputs > 0.5] = 1.0
            total_predictions += y.shape[0] * y.shape[1]
            correct_predictions += (pred.long() == y.long()).sum().item()

            running_loss += loss.item()

            optimizer.step()
            torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train Loss", loss.item(), niter)

                # metrics
                p, r, f1, _ = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy())
                micro = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy(),average='micro')
                macro = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy(),average='macro')
                
                acc = (correct_predictions / total_predictions)
                pbar.set_postfix(curr_loss=round(loss.item(), 4),
                                 acc_avg=round(acc, 4),
                                 f1=round(micro[2], 4)
                                 )

                pbar.update(10 if pbar.n + 50 <= pbar.total else pbar.total - pbar.n)

    running_loss /= len(train_loader)

    return running_loss, acc, p, r, f1, micro, macro


def validate(loader):
    global net, optimizer, criterion
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        preds = []

        for batch_idx, (x, seq_len, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda().double()
            outputs = net(x, seq_len)
            outputs = outputs.cuda().double()
            y = y.cuda().double()
            loss = criterion(outputs, y).detach()

            pred = torch.zeros(outputs.shape).cuda()
            pred[outputs > 0.5] = 1.0
            preds += list(pred.cpu().numpy())
            total_predictions += y.shape[0] * y.shape[1]
            correct_predictions += (pred.long() == y.long()).sum().item()

            running_loss += loss.item()

        running_loss /= len(val_loader)
        acc = (correct_predictions / total_predictions)
        p, r, f1, _ = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy())
        micro = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy(),average='micro')
        macro = precision_recall_fscore_support(y.cpu().numpy(), pred.cpu().numpy(),average='macro')

        return running_loss, acc, p, r, f1, micro, macro, np.array(preds)


def evaluate(p, r, f1, f1_micro, f1_macro, dataset):
    test_p_micro, test_r_micro, test_f1_micro, _ = f1_micro
    test_p_macro, test_r_macro, test_f1_macro, _ = f1_macro
    
    metrics = get_metrics_df()
    for i in range(metrics.shape[0]):
        metrics.iloc[i] = p[i], r[i], f1[i]
    metrics.loc["micro_avg"] = np.array([test_p_micro, test_r_micro, test_f1_micro])
    metrics.loc["macro_avg"] = np.array([test_p_macro, test_r_macro, test_f1_macro])
    metrics = metrics.round(3)
    metrics.to_csv("result/%s_%s.csv" % (model_stamp, dataset.name))
    return metrics

def run_epochs():
    epoch = 0
    if args.resume:
        check_point = torch.load(args.resume)
        epoch = check_point["epoch"] + 1
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
    print("statistics for epoch are average among samples and micro average among classes if possible")
    best_val_f1_micro = 0
    for e in range(epoch, args.epoch):
        if args.annealing:
            scheduler.step()

        train_loss, train_acc, train_p, train_r, train_f1, train_micro, train_macro = train(epoch, writer)
        val_loss, val_acc, val_p, val_r, val_f1, val_micro, val_macro, val_preds = validate(val_loader)
        test_loss, test_acc, test_p, test_r, test_f1, test_micro, test_macro, test_preds = validate(test_loader)

        train_p_micro, train_r_micro, train_f1_micro, _= train_micro
        val_p_micro, val_r_micro, val_f1_micro, _ = val_micro
                                        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("\re %3d: Train:l:%.3f|acc:%.3f|f1:%.3f|||Val:l:%.3f|acc:%.3f|f1:%.3f" %
              (e, train_loss, train_acc, train_f1_micro,
               val_loss, val_acc, val_f1_micro))

        # save check point
        if not args.debug and val_f1_micro > best_val_f1_micro:
            best_val_f1_micro = val_f1_micro
            # save model
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)
            # evaluate model only for the best epoch
            _ = evaluate(test_p, test_r, test_f1, test_micro, test_macro, dataset=test_dataset)
            np.save("result/%s_pred.npy" % model_stamp, np.array(val_preds))

    writer.close()
    #print("predicting result on test dataset...")
    #test_loss, test_acc, test_p, test_r, test_f1_micro, test_f1_macro, preds = validate(test_loader)
    #print("T:l:%.3f|acc:%.3f" % (test_loss, test_acc))
    #metrics = evaluate(test_p, test_r, test_f1, dataset=test_dataset)
    #print(metrics)



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
    # criterion = nn.BCELoss()  # Binary Cross Entropy loss with Sigmoid
    class_weight = torch.from_numpy(1 / train_dataset.proportion).cuda().double()
    criterion = nn.MultiLabelSoftMarginLoss(weight=class_weight)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

    if args.predict:
        check_point = torch.load(args.predict)
        net.load_state_dict(check_point["net"])
        test_loss, test_acc, test_p, test_r, test_f1, test_micro, test_macro, test_preds = validate(test_loader)
        metrics = evaluate(test_p, test_r, test_f1, test_micro, test_macro, dataset=test_dataset)
    else:
        run_epochs()
