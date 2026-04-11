import os
import random
import pickle
import h5py
import numpy as np
from typing import Sequence
import torch
import torch.utils.data as data
import myesm
import pandas as pd

class SeqPair(data.Dataset):
    def __init__(self, root,  split='test'):
        self.root = root
        self.datapath = os.path.join(self.root, 'pairs', split + '.tsv')
        self.protein_dict = self.read_fasta(os.path.join(self.root, 'seqs', split + '.fasta'))
        self.Seq_Name = list(self.protein_dict.keys())
        self.Sequence = list(self.protein_dict.values())

        with open(self.datapath, 'r') as f:
            self.samples = [line.strip().split('\t') for line in f.readlines()]

        print('Number of protein complexs:', len(self.samples))
        self.alphabet = myesm.Alphabet.default_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.max_parter_len = 510
        self.max_single_len = 1022

    def __getitem__(self, index):
        fpro, spro, label = self.samples[index]
        seq_A = self.protein_dict[fpro][:self.max_single_len]
        seq_B = self.protein_dict[spro][:self.max_single_len]
        complex_seq_A = seq_A[: self.max_parter_len] + '.' + seq_B[: self.max_parter_len]
        complex_len = len(complex_seq_A)
        complex_name = (fpro + '-' + spro).encode()
        masked_tokens_A, masked_pos_A, batch_tokens_A = self.batch_converter([('seq', seq_A)])
        masked_tokens_B, masked_pos_B, batch_tokens_B = self.batch_converter([('seq', seq_B)])
        masked_tokens_C, masked_pos_C, batch_tokens_C = self.batch_converter([('seq', complex_seq_A)])

        batch_tokens_A = np.array(batch_tokens_A)
        masked_tokens_A = np.array([[0 for i in range(len(seq_A) + 2)]])
        masked_pos_A = np.array([[0 for i in range(len(seq_A) + 2)]])

        batch_tokens_B = np.array(batch_tokens_B)
        masked_tokens_B = np.array([[0 for i in range(len(seq_B) + 2)]])
        masked_pos_B = np.array([[0 for i in range(len(seq_B) + 2)]])

        batch_tokens_C = np.array(batch_tokens_C)
        masked_tokens_C = np.array([[0 for i in range(complex_len + 2)]])
        masked_pos_C = np.array([[0 for i in range(complex_len + 2)]])

        batch_tokens = {'chain_A': batch_tokens_A[0], 'chain_B': batch_tokens_B[0], 'complex': batch_tokens_C[0]}
        masked_tokens = {'chain_A': masked_tokens_A[0], 'chain_B': masked_tokens_B[0], 'complex': masked_tokens_C[0]}
        masked_pos = {'chain_A': masked_pos_A[0], 'chain_B': masked_pos_B[0], 'complex': masked_pos_C[0]}
        protein_len = {'chain_A': np.array([len(seq_A)]), 'chain_B': np.array([len(seq_B)]), 'complex': np.array([complex_len])}
        return batch_tokens, masked_tokens, masked_pos, protein_len, complex_name

    def __len__(self):
        return len(self.samples)

    def read_fasta(self, file_path):
        Seq_dict = {}
        name = ''
        seq = ''
        with open(file_path, 'r') as f:
            for l in f.readlines():
                if l.startswith('>'):
                    if len(name) != 0 :
                        Seq_dict[name] = seq
                        seq = ''
                        name = ''
                    name = l.strip()[1:]
                else:
                    seq += l.strip().upper().replace(' ', '')
        Seq_dict[name] = seq
        return Seq_dict

class PPIDataset(data.Dataset):
    def __init__(self, split, args):
        self.split = split
        with open(os.path.join(args.data_dir, 'pairs', split + '.tsv'), 'r') as f:
                self.samples = [line.strip().split("\t") for line in f.readlines()]
        emb_path = os.path.join(args.data_dir, 'emb', split, 'embeddings.h5')

        self.proteins = set([sample[0] for sample in self.samples]) | set([sample[1] for sample in self.samples])
        emb_all_hf = h5py.File(emb_path, 'r')
        self.names = [n.decode() for n in list(emb_all_hf.get('Seq_Name'))]
        self.embs_all = np.array(emb_all_hf.get('embeddings'))
        self.seq_len_all = np.array(emb_all_hf.get('seq_len'))

    def get_embed(self, pro):
        if pro in self.names:
            pro_idx = self.names.index(pro)
        elif pro[:-1] in self.names:
            pro_idx = self.names.index(pro[:-1] )
        elif pro[:-4] in self.names:
            pro_idx = self.names.index(pro[:-4] )
        else:
            raise ValueError(f"{pro} embs not found!")

        emb = self.embs_all[pro_idx]

        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)
        pro_len = self.seq_len_all[pro_idx]
        return emb, pro_len

    def get_emb_dim(self):
        return self.embs_all.shape[1]

    def __getitem__(self, index):
        try:
            fpro, spro, label = self.samples[index]
        except ValueError:
            fpro, spro = self.samples[index]
            label = 0
        
        complex_name = fpro + '-' + spro
        pro_emb, pro_len = self.get_embed(complex_name)
        return torch.tensor(pro_emb[0]), torch.tensor(np.array(int(float(label))))

    def __len__(self):
        return len(self.samples)

class ScanDataset(data.Dataset):
    def __init__(self, split, args):
        self.split = split
        with open(os.path.join(args.data_dir, "pairs", split + ".tsv"), "r") as f:
            self.samples = [line.strip().split("\t") for line in f.readlines()]
        self.proteins = set([sample[0] for sample in self.samples]) | set([sample[1] for sample in self.samples])
        emb_path = os.path.join(args.data_dir, 'emb', split, 'embeddings.h5')
        emb_all_hf = h5py.File(emb_path, 'r')
        self.names = [n.decode() for n in list(emb_all_hf.get('Seq_Name'))]
        self.embs_all = np.array(emb_all_hf.get('embeddings'))
        self.seq_len_all = np.array(emb_all_hf.get('seq_len'))


    def get_embed(self, pro):
        if pro in self.names:
            pro_idx = self.names.index(pro)
        elif pro[:-1] in self.names:
            pro_idx = self.names.index(pro[:-1] )
        elif pro[:-4] in self.names:
            pro_idx = self.names.index(pro[:-4] )
        else:
            raise ValueError(f"{pro} embs not found!")

        emb = self.embs_all[pro_idx]

        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)
        pro_len = self.seq_len_all[pro_idx]
        return emb, pro_len

    def get_emb_dim(self):
        return self.embs_all.shape[1]

    def __getitem__(self, index):
        try:
            fpro, spro, label = self.samples[index]
        except ValueError:
            fpro, spro = self.samples[index]
            label = 0
        complex_name = fpro + '-' + spro
        pro_emb, pro_len = self.get_embed(complex_name)

        return torch.tensor(pro_emb[0]), fpro, spro, torch.tensor(np.array(int(float(label))))

    def __len__(self):
        return len(self.samples)


class SeqSingle(data.Dataset):
    def __init__(self,
                 root, split='test'):
        self.root = root
        self.protein_dict = {}
        
        all_dict = self.read_fasta(os.path.join(self.root, 'seqs', split + '.fasta'))
        self.datapath = os.path.join(self.root, 'pairs', split + '.tsv')
        df = pd.read_csv(self.datapath, sep = '\t', header = None, names = ['p1', 'p2', 'label'])
        unique_proteins = set(df['p1'].values.tolist()) | set(df['p2'].values.tolist())
        for k,v in all_dict.items():
            if k in unique_proteins:
                self.protein_dict[k] = v

        self.Seq_Name = list(self.protein_dict.keys())
        self.Sequence = list(self.protein_dict.values())
        print('Number of unique proteins:', len(self.protein_dict))
        self.alphabet = myesm.Alphabet.default_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()
        #self.batch_converter = self.alphabet.get_traindata_converter()
        self.max_len = 1022

    def __getitem__(self, index):
        seq_name = self.Seq_Name[index].encode()
        protein_seq = self.Sequence[index][:self.max_len]
        masked_tokens_A, masked_pos_A, batch_tokens_A = self.batch_converter([('seq', protein_seq)])
        batch_tokens_A = np.array(batch_tokens_A)
        masked_tokens_A = np.array([[0 for i in range(len(protein_seq) + 2)]])
        masked_pos_A = np.array([[0 for i in range(len(protein_seq) + 2)]])
        batch_tokens = {'chain_A': batch_tokens_A[0], 'chain_B': batch_tokens_A[0], 'complex' : batch_tokens_A[0]}
        masked_tokens = {'chain_A': masked_tokens_A[0], 'chain_B': masked_tokens_A[0], 'complex' : masked_tokens_A[0]}
        masked_pos = {'chain_A': masked_pos_A[0], 'chain_B': masked_pos_A[0], 'complex' : masked_pos_A[0]}
        protein_len = {'chain_A': np.array([len(protein_seq)]), 'chain_B': np.array([len(protein_seq)]), 'complex' : np.array([len(protein_seq)])}
        return batch_tokens, masked_tokens, masked_pos, protein_len, seq_name


    def __len__(self):
        return len(self.Sequence)

    def read_fasta(self, file_path):
        Seq_dict = {}
        name = ''
        seq = ''
        with open(file_path, 'r') as f:
            for l in f.readlines():
                if l.startswith('>'):
                    if len(name) != 0 :
                        Seq_dict[name] = seq
                        seq = ''
                        name = ''
                    name = l.strip()[1:]
                else:
                    seq += l.strip().upper().replace(' ', '')
        Seq_dict[name] = seq
        return Seq_dict
