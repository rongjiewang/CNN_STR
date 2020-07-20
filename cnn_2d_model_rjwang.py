#encoding=utf-8
#https://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/  连接两个输入例子
from __future__ import print_function
from keras.callbacks import LambdaCallback, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l1
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys
import io
import math
import keras
from keras.models import load_model
from keras import backend as K
from itertools import product
from Bio import SeqIO
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
import time
import data_process as load_data
#np.random.seed(1337) # for reproducibility
# Settings
ltrdict = {'a':[1,0,0,0],
           'c':[0,1,0,0],
           'g':[0,0,1,0],
           't':[0,0,0,1],
           'n':[0,0,0,0],
           'A':[1,0,0,0],
           'C':[0,1,0,0],
           'G':[0,0,1,0],
           'T':[0,0,0,1],
           'N':[0,0,0,0]}

chars = "ACGT"
print('total chars:', len(chars))
print('chars:', chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
input_dim = len(chars)

def encode_fasta_sequences(fname):
    """
    One hot encodes sequences in a  fasta file 
    """
    name, seq_chars = None, []
    sequences = []
    with open(fname, 'rb') as fp:
        data=str(fp.read()).strip().split('\n')
    
    for line in data:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                sequences.append(''.join(seq_chars).upper())
            name, seq_chars = line, []
        else:
            seq_chars.append(line)
    if name is not None:
        sequences.append(''.join(seq_chars).upper())
    return one_hot_encode(np.array(sequences))

def encode_fasta_gzipped_sequences(fname):
    """
    One hot encodes sequences in a gzipped fasta file 
    """
    import gzip
    name, seq_chars = None, []
    sequences = []
    with gzip.open(fname, 'rb') as fp:
        data=str(fp.read()).strip().split('\n')
    
    for line in data:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                sequences.append(''.join(seq_chars).upper())
            name, seq_chars = line, []
        else:
            seq_chars.append(line)
    if name is not None:
        sequences.append(''.join(seq_chars).upper())
    return one_hot_encode(np.array(sequences))

def one_hot_encode(seqs):
    encoded_seqs=np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
    encoded_seqs=np.expand_dims(encoded_seqs,1)
    return encoded_seqs

def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    seqs = []
    labels = []
    for record in records:
        seqs.append(str(record.seq).upper())
        labels.append(int(record.id))
    return seqs, labels

def vectorization(seqs, labels):
    kmer = np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
    kmer = np.expand_dims(kmer,1)
    return kmer, labels
def get_batch(seqs, labels):
    duration = len(labels)
    for i in range(0,duration//batch_size):
        idx = i*batch_size
        yield vectorization(seqs[idx:idx+batch_size],labels[idx:idx+batch_size])
def get_random_batch(seqs, labels):
    duration = len(seqs)-1
    for i in range(0,duration//batch_size):
        seq = []
        label=[]
        for j in range(batch_size):
            k = random.randint(0, duration)
            seq.append(seqs[k])
            label.append(labels[k])
        yield vectorization(seq, label)

def my_kernel_initializer(shape, dtype=None):
    x = np.zeros(shape, dtype=np.bool)
    for i, c in enumerate(product('ACGT', repeat=5)):
        kmer=c*3
        for t, char in enumerate(kmer):
            x[t,char_indices[char],i] = 1
    return x

def loadModel():
    #model.load_weights('my_model_weights.h5')
    #json and create model
    json_file = open('../model/mit/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("../model/mit/model.h5")
    print("Loaded model from disk")
    return model


def model_patern():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     #kernel_initializer=my_kernel_initializer,
                     #trainable=False,
                     #padding='same',
                     #activation=None,
                     #use_bias=False,
                     #bias_initializer= keras.initializers.Constant(value=-7),
                     strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
def model_CNN_LSTM_2D():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv2D(
        filters=4096, kernel_size=(8,input_dim), strides=(1, 1), padding='valid', 
        data_format='channels_first', 
        dilation_rate=1, 
        activation='relu', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None,
        input_shape=(1,maxlen,input_dim)))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(MaxPooling2D(pool_size=(6, 1)))

    model.add(Conv2D(
        filters=1024, kernel_size=(6,1), strides=(1, 1), padding='valid', 
        data_format='channels_first', 
        dilation_rate=1, 
        activation='relu', 
        use_bias=True, 
        kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))#Sigmoid
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return model

def model_CNN_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,
                     input_shape=(batch_size,4)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(input_dim))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def model_CNN():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(Conv1D(filters=320,
                     kernel_size=6,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,
                     input_shape=(maxlen,input_dim)))
    model.add(MaxPooling1D(pool_size=3,strides=3))
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=480,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=960,
                     kernel_size=4,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
def model_LSTM():
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(input_dim, activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def saveModel(epoch):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../saved_model/model.json", "w") as json_file:
        json_file.write(model_json)
    #serialize weights to HDF5
    name="../saved_model/model_"+str(epoch)+".h5"
    model.save_weights(name)
    print("Saved model to disk")
    return

def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print('----- Testing entorpy after Epoch: %d' % (epoch+1))
    accuracy = 0
    batch_num = 0
    seqs,labels = read_fasta(test_path)
    for i, batch in enumerate(get_batch(seqs,labels)):
        _input = batch[0]
        _labels = batch[1]
        x=model.test_on_batch(_input,_labels)
        accuracy += x[1]
        batch_num += 1
    return accuracy/batch_num
def logging(s, log_path, print_=False, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

if __name__ == '__main__':
    batch_size = 64
    epochs = 10
    maxlen = 128
    train_path = '../data/generate_STR/total_train.fasta'
    test_path = '../data/generate_STR/total_test.fasta'
    log_path = '../data/log.txt'
    total_start_time = time.time()
    model = model_CNN_LSTM_2D()
    #model = model_CNN()
    print(model.summary())
    accuracy = []
    for epoch in range(epochs):
        seqs,labels = read_fasta(train_path)
        total_num_batch = len(labels)//batch_size
        for i, batch in enumerate(get_batch(seqs,labels)):
            start_time = time.time()
            _input = batch[0]
            _labels = batch[1]
            score_eval=model.train_on_batch(_input,_labels)
            elapsed = time.time() - start_time
            log_str = '| epoch: {:3d} | batche/batches: {:>6d}/{:d} ' \
                      '| s/batch: {:5.2f} | loss: {:5.4f} | accuracy: {:5.4f}'.format(
                epoch+1,  i, total_num_batch, elapsed, score_eval[0], score_eval[1])
            logging(log_str,log_path)
            if(i%100==0):
                # print("epoch:",epoch,'\t',
                # 	  "batch:",'\t',i,'\t', 
                # 	  model.metrics_names[0], ' : ',score_eval[0],'\t',
                #       model.metrics_names[1], ' : ', '\t',score_eval[1])
                print('| epoch: {:3d} | batche/batches: {:>6d}/{:d} ' \
                      '| s/batch: {:5.2f} | loss: {:5.4f} | accuracy: {:5.4f}'.format(
                epoch+1,  i, total_num_batch, elapsed, score_eval[0], score_eval[1]))
        saveModel(epoch+1)
        test_accuracy = on_epoch_end(epoch)
        print('Test accuracy is {:5.4f}'.format(test_accuracy))
        accuracy.append(test_accuracy)
    total_elapsed_time = time.time() -total_start_time
    print('Test accuracy for epochs:',accuracy)
    print('Total time used is {:5.2f} s'.format(total_elapsed_time))
    
