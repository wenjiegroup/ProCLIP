import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
from torch import nn
from utils.dataset import PPIDataset
from models.classifier import PPIClassifier
import numpy as np
import shutil
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score, auc
from math import cos, pi

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='examples/human', help='path to dataset')
    parser.add_argument('--trainsplit', default='test', help='path to dataset')
    parser.add_argument('--testsplit', default='test', help='path to dataset')
    parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--learning_rate', default=0.00003, type=float, help='learning rate in training')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--interval_to_val', type=int, default=1, help='number of epochs to run validation')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=150, help='manual seed')
    parser.add_argument('--init_weight', type=bool, default=True, help='init the weight of the network')
    parser.add_argument('--lr_decay_interval', type=int, default=1)
    parser.add_argument('-s','--save_model', default='Trained_model/classifier.pth', help='save model')
    parser.add_argument('-t', '--task', default='dscript')
    opt = parser.parse_args()

    niter = 0

    dset = PPIDataset(split = opt.trainsplit, args = opt)
    test_dset = PPIDataset(split = opt.testsplit, args = opt)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pointcls_net = PPIClassifier(dset.get_emb_dim(), [2048, 1024], batch_norm=True, dropout=0.05)

    if opt.init_weight:
        for m in pointcls_net.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                m.weight.data.normal_(mean = 0, std = 0.01)
                m.bias.data.fill_(0.0)

    if opt.cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        pointcls_net = torch.nn.DataParallel(pointcls_net)
        pointcls_net.to(device)


    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    setup_seed(opt.manualSeed)

    optimizer = torch.optim.Adam(pointcls_net.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-05,
                                  weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss().to(device)


    best_metrics = -0.1
    lr_max = opt.learning_rate
    lr_min = 0.000001
    sig = nn.Sigmoid()
    lr = opt.learning_rate
    for epoch in range(opt.niter):
        if epoch > 0 and epoch % opt.lr_decay_interval == 0:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for i, data in enumerate(dataloader, 0):
            features, labels = data
            labels = labels.to(device)
            
            niter = epoch * len(dataloader) + i
            pointcls_net.train()
            pointcls_net.zero_grad()
            optimizer.zero_grad()
           
            features = features.to(device)
            output = pointcls_net(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if(i == 0):
                print('[%d/%d][%d/%d] Loss: %.4f '
                      % (epoch, opt.niter, i, len(dataloader), loss ))

        scheduler.step()

        if (epoch % opt.interval_to_val == 0):
            label_all = []
            prediciton_all = []
            y_score = []
            print('Test once')
            for i, data in enumerate(test_dataloader, 0):
                features, labels = data
                labels = labels.to(device)
                features = features.to(device)
                pointcls_net.eval()
                output = pointcls_net(features)
                _,predictions = torch.max(output.data, dim = 1)
                if(i  == 0):
                    print('[%d/%d][%d/%d] '% (epoch, opt.niter, i, len(test_dataloader)))
                #f1 score
                if (len(label_all) == 0):
                    label_all = labels.detach().cpu().numpy()
                    prediciton_all = predictions.detach().cpu().numpy()
                    y_score = output[: , 1].detach().cpu().numpy()
                else:
                    label_all = np.concatenate((label_all, labels.detach().cpu().numpy()))
                    prediciton_all = np.concatenate((prediciton_all, predictions.detach().cpu().numpy()))
                    y_score = np.concatenate((y_score, output[: , 1].detach().cpu().numpy()))

            acc = accuracy_score(label_all, prediciton_all)
            precision, recall, _ = precision_recall_curve(label_all, y_score, pos_label=1)
            auprc = auc(recall, precision)
            print('accuracy: %.4f'%acc, 'AUPR: %.4f'%auprc)

            score = auprc
            if (score > best_metrics):
                best_metrics = score
                print('The best auprc occurs in epoch %d: %.4f' % (epoch, score))
                torch.save({'epoch': epoch + 1, 'state_dict': pointcls_net.state_dict()},
                       opt.save_model)


if __name__ == '__main__':
    main()
