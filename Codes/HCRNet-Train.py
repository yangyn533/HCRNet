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

from attention import Attention

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
        else:
            continue        
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


def build_class_file(np, ng, class_file):
    with open(class_file, 'w') as outfile:
        outfile.write('label' + '\n')
        for i in range(np):
            outfile.write('1' + '\n')
        for i in range(ng):
            outfile.write('0' + '\n')
            
            

def circRNA2Vec(k, s, vector_dim, model, MAX_LEN, pos_sequences, neg_sequences):   
    model1 = gensim.models.Doc2Vec.load(model)
    pos_list = seq2ngram(pos_sequences, k, s, model1.wv)
    neg_list = seq2ngram(neg_sequences, k, s, model1.wv)
    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=MAX_LEN,padding='post')

    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    y = to_categorical(y)


    indexes = np.random.choice(len(y), len(y), replace=False)
    dataX = np.array(X)[indexes]
    dataY = np.array(y)[indexes]

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    print(embedding_matrix.shape)
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector    

    return X, y, embedding_matrix

    
    
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


def Generate_Embedding(seq_posfile, seq_negfile, model):
    
    seqpos = read_fasta_file(seq_posfile)
    seqneg = read_fasta_file(seq_negfile)
        
    X, y, embedding_matrix = circRNA2Vec(10, 1, 30, model, 101, seqpos, seqneg)
    return X, y, embedding_matrix


def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)

def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]


def createModel(embedding_matrix):
    sequence_input = Input(shape=(101, ), name='sequence_input')
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(sequence_input)
    sequence = Convolution1D(filters=128, kernel_size=1, padding='same')(embedding)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('relu')(sequence)
    profile_input = Input(shape=(101,768), name='profile_input')
    profile = Convolution1D(filters=128, kernel_size=1, padding='same')(profile_input)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('relu')(profile)
    property_input = Input(shape=(101,99), name='property_input')
    propertys = Convolution1D(filters=128, kernel_size=1, padding='same')(property_input)
    propertys = BatchNormalization(axis=-1)(propertys)
    propertys = Activation('relu')(propertys)
    mergeInput = Concatenate(axis=-1)([sequence, profile, propertys])
    overallResult = TCN(nb_filters=64, kernel_size=10, dropout_rate=0.3, nb_stacks=1,  dilations=[1, 2, 4, 8, 16, 32, 64], return_sequences=True,activation='relu',padding='same',use_skip_connections=True)(mergeInput)
    overallResult = (GlobalExpectationPooling1D(mode = 0, m_trainable = False, m_value = 1))(overallResult)
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)    
    return Model(inputs=[sequence_input, profile_input, property_input], outputs=[ss_output])


def parse_arguments(parser):
    parser.add_argument('--RBP_ID', type=str)
    parser.add_argument('--kmer', type=int,  default=3)
    parser.add_argument('--modelType', type=str, default='~./modelpath/circRNA2Vec_model', help='generate the embedding_matrix')
    parser.add_argument('--storage', type=str, default='~./resultpath/result/')
    args = parser.parse_args()
    return args   
    
def main(parser):
    protein = parser.RBP_ID
    k = parser.kmer
    model = parser.modelType
    file_storage = parser.storage

    seqpos_path = '~./datapath/' + protein + '/positive'
    seqneg_path = '~./datapath/' + protein + '/negative'

  
    Property = dealwithdata(protein)
    Embedding = circRNABert(protein,k)
    Kmer, dataY,  embedding_matrix = Generate_Embedding(seqpos_path, seqneg_path, model)     
    
    indexes = np.random.choice(Kmer.shape[0],Kmer.shape[0], replace=False)
    
    
    training_idx, test_idx = indexes[:round(((Kmer.shape[0])/10)*8)], indexes[round(((Kmer.shape[0])/10)*8):]
    
    train_property, test_property = Property[training_idx, :, :], Property[test_idx, :, :]
    train_sequence, test_sequence = Kmer[training_idx, :], Kmer[test_idx, :]
    train_profile, test_profile = Embedding[training_idx, :, :], Embedding[test_idx, :, :]
    print(train_profile.shape)
    train_label, test_label = dataY[training_idx, :], dataY[test_idx, :] 
    
    batchSize = 20
    maxEpochs = 100
    basic_path = file_storage  + '/'
    methodName = protein

    logging.basicConfig(level=logging.DEBUG)
    sys.stdout = sys.stderr
    logging.debug("Loading data...")

    
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
    
    
    test_y = test_label[:, 1]

    kf = KFold(5, True)
    aucs = []
    Acc = []
    precision1 = []
    recall1 = []
    fscore1 = []
    i = 0
    
    for train_index, eval_index in kf.split(train_label):
        train_X1 = train_sequence[train_index]
        train_X2 = train_profile[train_index] 
        train_X3 = train_property[train_index] 
        train_y = train_label[train_index]
        eval_X1 = train_sequence[eval_index]
        eval_X2 = train_profile[eval_index]
        eval_X3 = train_property[eval_index] 
        eval_y = train_label[eval_index]        

        print ('training_network size is ', len(train_X1))
        print ('validation_network size is ', len(eval_X1))
    
    
        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path, methodName,
                                                                                     str(i))
        logging.debug("Loading network/training configuration...")
        model = createModel(embedding_matrix)
        logging.debug("Model summary ... ")
        model.count_params()
        model.summary()
        checkpoint_weight = CHECKPOINT_PATH + "weights.best.hdf5"
        if (os.path.exists(checkpoint_weight)):
            print ("load previous best weights")
            model.load_weights(checkpoint_weight)

        model.compile(optimizer='adam',
                      loss={'ss_output': 'categorical_crossentropy'},metrics = ['accuracy'])
        logging.debug("Running training...")

        def step_decay(epoch):
            initial_lrate = 0.0005
            drop = 0.8
            epochs_drop = 5.0            
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            print (lrate)
            return lrate

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto'),
            ModelCheckpoint(checkpoint_weight,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                            period=1),
            LearningRateScheduler(step_decay),
            TensorBoard(log_dir=LOG_PATH, histogram_freq=1, write_graph=True, write_images=True,
                                                            embeddings_freq = 1,
                                                            embeddings_layer_names = None,
                                                            embeddings_metadata = None
                        )
        ]
        startTime = time.time()
        history = model.fit(
            {'sequence_input': train_X1, 'profile_input': train_X2, 'property_input': train_X3},
            {'ss_output': train_y},
            epochs=maxEpochs,
            batch_size=batchSize,
            callbacks=callbacks,
            verbose = 1,
            validation_data=(
                {'sequence_input': eval_X1, 'profile_input': eval_X2, 'property_input': eval_X3},
                {'ss_output':eval_y}),
            shuffle=True)
        endTime = time.time()
        logging.debug("Saving final model...")
        model.save(os.path.join(MODEL_PATH, 'model.h5'), overwrite=True)
        json_string = model.to_json()
        with open(os.path.join(MODEL_PATH, 'model.json'), 'w') as f:
            f.write(json_string)
        logging.debug("make prediction")       
        ss_y_hat_test = model.predict(
                  {'sequence_input': test_sequence, 'profile_input': test_profile,'property_input': test_property}) 

        ytrue = test_y
        ypred = ss_y_hat_test[:, 1]
        
        y_pred = np.argmax(ss_y_hat_test, axis=-1)
        auc = roc_auc_score(ytrue, ypred)
        np.save(RESULT_PATH + 'auc.npy',auc)
        fpr,tpr,thresholds=roc_curve(ytrue,ypred)
        tprs.append(interp(mean_fpr,fpr,tpr))
        aucs.append(auc)        
        acc = accuracy_score(ytrue, y_pred)
        np.save(RESULT_PATH + 'acc.npy',acc)
        Acc.append(acc)
        
        precision = precision_score(ytrue, y_pred)  
        recall = recall_score(ytrue, y_pred)
        fscore = f1_score(ytrue, y_pred)         
        precision1.append(precision)
        recall1.append(recall)
        fscore1.append(fscore)     
        i = i + 1
    

    print("acid AUC: %.4f " % np.mean(aucs))
    print("acid ACC: %.4f " % np.mean(Acc))
    print("acid Precision: %.4f " % np.mean(precision1))
    print("acid Recall: %.4f " % np.mean(recall1))

    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc= np.mean(aucs)
    mean_acc = np.mean(Acc)
    mean_precision = np.mean(precision1)
    mean_recall = np.mean(recall1)
    mean_fscore = np.mean(fscore1)
    
    np.save(basic_path + methodName + '/' + 'mean_fpr.npy',mean_fpr)
    np.save(basic_path + methodName + '/' + 'mean_tpr.npy',mean_tpr)
    np.save(basic_path + methodName + '/' + 'mean_auc.npy',mean_auc)
    np.save(basic_path + methodName + '/' + 'mean_acc.npy',mean_acc)
    np.save(basic_path + methodName + '/' + 'mean_precision.npy',mean_precision)
    np.save(basic_path + methodName + '/' + 'mean_recall.npy',mean_recall)  
    np.save(basic_path + methodName + '/' + 'mean_fscore.npy',mean_fscore)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    main(args)