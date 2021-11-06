from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.layers import Reshape, Dense, Convolution1D, Dropout, Input, Activation, Flatten,MaxPool1D,add, AveragePooling1D, Bidirectional,GRU,LSTM,Multiply, MaxPooling1D,TimeDistributed,AvgPool1D
from keras.layers.merge import Concatenate,concatenate
from keras.layers.wrappers import Bidirectional
from six.moves import cPickle as pickle
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,RMSprop, Adamax, Nadam
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.regularizers import l2, l1
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.backend import sigmoid
from keras import metrics
from keras.constraints import max_norm
import logging
import os
import sys
import numpy as np
import time
import argparse
import math
import logging
import os
import sys
import numpy as np
import time
import math
import tensorflow as tf
import collections
from itertools import cycle
from scipy import interp

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Lambda
from keras.layers import dot
import sys
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical

import keras
import os
import pandas as pd
import numpy as np
import pickle
import pdb
import logging, multiprocessing
from collections import namedtuple
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from keras_self_attention import SeqSelfAttention,ScaledDotProductAttention
from keras_multi_head import MultiHead,MultiHeadAttention
from scipy import interp
import matplotlib.pyplot as plt
from FeatureEncoding import dealwithdata

from ePooling import *
from BertDealEmbedding import circRNABert
from tcn import TCN
from sklearn.decomposition import PCA
from matplotlib import pyplot

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


np.random.seed(4)

def seq2ngram(seqs, k, s, wv):
    list22 = []
    print('need to n-gram %d lines' % len(seqs))

    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line) 
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22
    
def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def split_overlap_seq(seq):
    window_size = 101
    overlap_size = 20
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - 101)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - 101)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        bag_seqs.append(seq)
    else:
        if remain_ins > 10:
            new_size = end - overlap_size
            seq1 = seq[-new_size:]
            bag_seqs.append(seq1)
    return bag_seqs

def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences):
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    seqs = pos_list

    X = pad_sequences(seqs, maxlen=MAX_LEN,padding='post')

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    
            
    return X, embedding_matrix
    
    
def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0]=='>': 
            name = line[1:] 
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)
        
    return np.asarray(bag_sen)


def Generate_Embedding(seqpos_path, model):
    
    seqpos = read_fasta_file(seqpos_path)
        
    X, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos)
    return X, embedding_matrix

def parse_arguments(parser):
    parser.add_argument('--RBP_ID', type=str)
    parser.add_argument('--seqPath', type=str,  default='~./datapath/InputData/')
    parser.add_argument('--modelType', type=str, default='~./modelpath/circRNA2Vec_model', help='generate the embedding_matrix')
    parser.add_argument('--Predictmodel_path', type=str, default='~./resultpath/result/')
    args = parser.parse_args()
    return args   
    
def main(parser):
    protein = parser.RBP_ID
    seq_path = parser.seqPath
    model = parser.modelType
    predictmodel_path = parser.Predictmodel_path

    predictmodel = predictmodel_path + protein + '/' + '/model/' + 'model.h5'
    
    listpath = os.listdir(seq_path)
    
    for each_item in listpath:
        seqpos_path = seq_path + each_item
        
        Kmer = dealwithdata(seqpos_path)
    
        
        EmbeddingData, embedding_matrix = Generate_Embedding(seqpos_path, model)    
        
        BertEmbedding = circRNABert(seqpos_path, 3)
    
        modelPredict = load_model(predictmodel,custom_objects={'TCN':TCN, 'GlobalExpectationPooling1D':GlobalExpectationPooling1D})
        predictedResult = modelPredict.predict([EmbeddingData, BertEmbedding,Kmer])
        predictedResult = predictedResult[:, 1]
        print(predictedResult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)