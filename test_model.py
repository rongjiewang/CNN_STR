from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.models import model_from_json
import numpy as np
import random
from keras.models import load_model
from itertools import product
from Bio import SeqIO
import time
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


def read_fasta(data_path):
    records = list(SeqIO.parse(data_path, "fasta"))
    seqs = []
    labels = []
    for record in records:
        seqs.append(str(record.seq))
        labels.append(int(record.id))
    return seqs, labels

def vectorization(seqs, labels):
    kmer = np.array([[ltrdict.get(x,[0,0,0,0]) for x in seq] for seq in seqs])
    kmer = np.expand_dims(kmer,1)
    #predict = np.squeeze(labels,1)
    return kmer, labels
def get_batch(seqs, labels):
    duration = len(seqs)
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


def loadModel(epoch):
    #model.load_weights('my_model_weights.h5')
    #json and create model
    json_file = open('../saved_model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    name="../saved_model/model_"+str(epoch)+".h5"
    model.load_weights(name)
    print("Loaded model from disk")
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

if __name__ == '__main__':
    batch_size = 16
    maxlen = 128
    test_path = '../data/generate_STR/human_chr1_test.fasta'
    #test_path = '../data/generate_STR/human_chr1_test.fasta'
    model = loadModel(7)
    #model = model_CNN()
    print(model.summary())
    seqs,labels = read_fasta(test_path)
    error_num = 0
    for i, batch in enumerate(get_batch(seqs,labels)):
        _input = batch[0]
        _labels = batch[1]
        y=model.predict_on_batch(_input)
        for i, (pre, label) in enumerate(zip(y,_labels)):
            #print(pre,label)
            if(pre > 0.5 and label==0):
                #print("False pos:",seqs[i],label)
                error_num +=1
            elif(pre < 0.5 and label==1):
                #print("False neg:",seqs[i],label)
                error_num +=1
    print('accuracy is {:5.2f}'.format((len(labels)-error_num)*100/len(labels)))
    print("finishing")   
