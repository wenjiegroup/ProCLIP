import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
from torch import nn
from utils.dataset import ScanDataset
from models.classifier import PPIClassifier
import numpy as np
import shutil
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score, auc
from sklearn.metrics import recall_score,accuracy_score,roc_curve
from sklearn.metrics import precision_recall_curve,PrecisionRecallDisplay
import  torch.nn.functional as F
import pandas as pd
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='examples/human', help='path to dataset')
    parser.add_argument('--testsplit', default='test', help='path to dataset')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('-m', '--load_model', default='Trained_model/classifier.pth')
    parser.add_argument('-f', '--outdata', default='results/interaction_score.csv')
    opt = parser.parse_args()

    test_dset = ScanDataset(split = opt.testsplit, args = opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pointcls_net = PPIClassifier(test_dset.get_emb_dim(), [2048, 1024], batch_norm=True, dropout=0.05)

    if opt.cuda:
        pointcls_net = torch.nn.DataParallel(pointcls_net)
        pointcls_net.to(device)
    pointcls_net.load_state_dict(torch.load(opt.load_model, weights_only=True)['state_dict'])
    pointcls_net.eval()


    all_test_splits = [opt.testsplit]
    start_time = datetime.datetime.now()
    for testsplit in all_test_splits:
        test_dset = ScanDataset(split = testsplit, args = opt)
        test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
        print('testsplit:', testsplit)
        fpro_all = []
        spro_all = []
        prediciton_all = []
        y_score = []
        for i, data in enumerate(test_dataloader, 0):
            features, fpro, spro = data
            features = features.to(device)
            output = pointcls_net(features, train=False)
            _,predictions = torch.max(output.data, dim = 1)
            logits = F.softmax(output)
            if (len(fpro_all) == 0):
                fpro_all = fpro
                spro_all = spro
                prediciton_all = predictions.detach().cpu().numpy()
                y_score = logits[: , 1].detach().cpu().numpy()
            else:
                fpro_all = np.concatenate((fpro_all, fpro))
                spro_all = np.concatenate((spro_all, spro))
                prediciton_all = np.concatenate((prediciton_all, predictions.detach().cpu().numpy()))
                y_score = np.concatenate((y_score, logits[: , 1].detach().cpu().numpy()))

        outdf = pd.DataFrame()
        outdf['p1'] = fpro_all
        outdf['p2'] = spro_all
        outdf['label'] = prediciton_all
        outdf['score'] = y_score
        outdf.to_csv(opt.outdata, index = False)

    current_time = datetime.datetime.now()
    print((current_time - start_time).total_seconds())

if __name__ == '__main__':
    main()
