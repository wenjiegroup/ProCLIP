import torch.optim as optim
import torch
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
import shutil
import numpy as np
import h5py
import myesm
from utils.dataset import SeqPair,SeqSingle


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def collate(samples):
    batch_sample = [{}, {}, {}, {}]
    traim_sample =  [{}, {}, {}, {}]
    batch_tokens, masked_tokens, masked_pos, protein_len, p_name = map(list, zip(*samples))

    for idx, item in enumerate([batch_tokens, masked_tokens, masked_pos, protein_len]):
        for key in ['chain_A', 'chain_B', 'complex']:
            for sub_item in item:
                if key not in batch_sample[idx].keys():
                    batch_sample[idx][key] = [sub_item[key]]
                else:
                    batch_sample[idx][key].append(sub_item[key])

    for idx in range(len(batch_sample)):
        for key in ['chain_A', 'chain_B', 'complex']:
            traim_sample[idx][key] = torch.tensor(np.array(batch_sample[idx][key]))
    return  traim_sample[0], traim_sample[1], traim_sample[2], traim_sample[3], np.array(p_name)

def init():

    logging.info(str(args))
    device = torch.device('cuda:0')

    dataset_single = SeqSingle(root=args.testdata, split = args.split)
    dataset_test = SeqPair(root=args.testdata, split = args.split)
    dataloader_single = torch.utils.data.DataLoader(dataset_single, batch_size=args.batch_size,
                                                  shuffle=False, collate_fn=collate, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, collate_fn=collate, num_workers=int(args.workers))
    logging.info('Length of test dataset:%d', len(dataset_test))
    model_module = importlib.import_module('.%s' % args.model_name, 'models')

    net = torch.nn.DataParallel(model_module.Model(args))
    net.to(device)

    if args.load_model:
        ckpt = torch.load(args.load_model, weights_only=True)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
        pretrained_dict = ckpt["net_state_dict"]

    val(net, dataloader_single, dataloader_test)

def val(net, dataloader_single, dataloader_test):
    logging.info('Testing...')
    device = torch.device('cuda:0')
    net.module.eval()
    n_count = 0
    total_emb = []
    file_name = []
    seq_len = []
    start_time = datetime.datetime.now()
    feat_dict = {}

    with torch.no_grad():
        for i, data in enumerate(dataloader_single):
            batch_tokens, masked_tokens, masked_pos, protein_len, p_name = data
            for item in [batch_tokens, masked_tokens, masked_pos, protein_len]:
                for k in item.keys():
                    item[k] = item[k].to(device)

            results = net(batch_tokens, masked_tokens, masked_pos, protein_len)
            batch_size = batch_tokens['complex'].size()[0]
            assert(batch_size == 1)
            for sample_id in range(batch_size):
                name = p_name.tolist()[sample_id].decode()
                feat_dict[name] = {}
                feat_dict[name]['mean'] = results['mean_rep']
                feat_dict[name]['all'] = results['all_feat']
                n_count += 1
                if(n_count % 1000 == 0):
                    logging.info('processing single seqs: %d : %d'%(n_count, len(dataloader_single)))

    n_count = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            batch_tokens, masked_tokens, masked_pos, protein_len, p_name = data
            for item in [batch_tokens, masked_tokens, masked_pos, protein_len]:
                for k in item.keys():
                    item[k] = item[k].to(device)

            batch_size = batch_tokens['complex'].size()[0]
            input_feat = {}
            p_names = p_name.tolist()
            for sample_id in range(batch_size):
                partners = p_names[sample_id].decode().split('-')
                input_feat['chain_A_mean'] = feat_dict[partners[0]]['mean']
                input_feat['chain_A_all'] = feat_dict[partners[0]]['all']
                input_feat['chain_B_mean'] = feat_dict[partners[1]]['mean']
                input_feat['chain_B_all'] = feat_dict[partners[1]]['all']

            for k in input_feat.keys():
                    input_feat[k] = input_feat[k].to(device)

            embedding = net(batch_tokens, masked_tokens, masked_pos, protein_len, result_dict = input_feat, compute_single = False)
            batch_size = batch_tokens['complex'].size()[0]
            assert(batch_size == 1)
            for sample_id in range(batch_size):
                length = protein_len['complex'][sample_id].item()
                name = p_names[sample_id].decode()
                emb = embedding.cpu().detach().numpy()
                total_emb.append(emb[0])
                file_name.append(name.encode('utf-8'))
                seq_len.append(length)
                n_count += 1
                if(n_count % 1000 == 0):
                    logging.info('processing complex seqs: %d : %d'%(n_count, len(dataloader_test)))

                
    current_time = datetime.datetime.now()
    print('Inference ', len(seq_len), ' pairs, ', (current_time - start_time).total_seconds())
    write_data(total_emb, file_name, seq_len)


def write_data(total_emb, p_name, p_len):
    if args.outfile != None:
        fname = args.outfile
    else:
        fname = 'embeddings.h5'
    
    f = h5py.File(fname, 'w')
    print(np.array(total_emb, dtype='float32').shape)
    f.create_dataset('embeddings', data=np.array(total_emb, dtype='float32'), compression='gzip', chunks=True, maxshape=(None,total_emb[0].shape[-1])) #2560 2688 3840
    f.create_dataset('Seq_Name', data=np.array(p_name), compression='gzip', chunks=True, maxshape=(1000000)) 
    f.create_dataset('seq_len', data=np.array(p_len), compression='gzip', chunks=True, maxshape=(1000000)) 
    f.close()


if __name__ == '__main__':
    config_path = 'cfgs/inference_emb.yaml'
    args = munch.munchify(yaml.safe_load(open(config_path)))
    time = datetime.datetime.now().isoformat()[:19]
    logging.Formatter.converter = beijing

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    init()
