
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
import pickle
from tqdm.auto import tqdm
import tensorflow as tf

def read_fasta(file_path):
    #dict = {}
    seq_list = []
    f = open(file_path,'r')
    for line in f:
        if '>' not in line:
            line = line.strip().upper()
            seq_list.append(line)
    return seq_list

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def circRNA_Bert(sequences, dataloader):
    features = []
    seq = []    
    tokenizer = BertTokenizer.from_pretrained("~.//", do_lower_case=False)
    model = BertModel.from_pretrained("/home/wangyansong/DNABERT/3-new-12w-0/")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)
    
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
        #print(ids)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        #print(attention_mask)
        with torch.no_grad():
            #embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[1]
            embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        embedding = embedding.cpu().numpy()
    
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            #print(embedding[0].shape)
            #seq_emd = embedding[seq_num][0:seq_len]
            #seq_emd = seq_emd.mean(0)
            seq_emd = embedding[seq_num][1:seq_len-1]
            #print(seq_emd)
            features.append(seq_emd) 
    return features
    


def circRNABert(protein,k):
    file_positive_path = '~./datapath/' + protein + '/positive'
    file_negative_path = '~./datapath/' + protein + '/negative'
    sequences_pos = read_fasta(file_positive_path)
    sequences_neg = read_fasta(file_negative_path)
    sequences_ALL = sequences_pos + sequences_neg
    sequences = []
    Bert_Feature = []  
    for seq in sequences_ALL:
        seq = seq.strip()
        seq_parser = seq2kmer(seq, k)
        #sequences.append(re.sub(r"[UZOB]", "X"," ".join(re.findall(".{1}",i.upper()))))
        sequences.append(seq_parser)
    #print(sequences)
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=100, shuffle=False)
    Features = circRNA_Bert(sequences, dataloader)
    #print(Features)
    #print(len(Features))
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature.tolist())
    arrayBF = np.array(Bert_Feature)
    data = np.pad(arrayBF, ((0,0),(0,2),(0,0)), 'constant', constant_values=0)
    return  data
