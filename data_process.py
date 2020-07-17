import os
import pandas as pd
import random
import numpy as np
from copy import deepcopy
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys


# read reference sequence data
def read_fasta():
    data_path = os.path.abspath(os.path.join(os.getcwd(), "../data/ref/ASM522150v1.fna"))
    records = list(SeqIO.parse(data_path, "fasta"))
    seqence = ""
    seq_len_list = []
    for record in records:
        seqence += str(record.seq)
        seq_len_list.append(len(record.seq))
    return records, seqence, seq_len_list

# read STR csv data
def get_STR_csv():
    test_path = os.path.abspath(os.path.join(os.getcwd(), "../data/tsv/Escherichia coli.tsv"))
    data = pd.read_csv(test_path, sep='\t', header=None,
                       names=['Chromosome', 'Start', 'Stop', 'Repeat Class', 'Repeat Length', 'Strand', '# Units',
                              'Actual Repeat', 'Gene Name', 'Gene Start', 'Gene Stop', 'Gene Strand', 'Annotation',
                              'Distance from TSS'])
    max_len = data['Repeat Length'].max()
    print("max STR length: ", max_len)
    max_len = max([len(x) for x in data['Actual Repeat']])
    print("max Unit length: ",max_len)
    return data

def get_data(length):
    STR_data = get_STR_csv()
    ref_records, ref_seq, ref_len_list = read_fasta()
    # initialization
    bz = []
    final_data = []
    final_label = []
    final_description = []

    for i in range(0, int(len(ref_seq) / length)):
        bz.append(0)
    # generate positive data
    num_positive = 0
    print("The number of STR is:", len(STR_data))
    # print("The length of referecen seq is: ", len(ref_seq))
    chromeName = []
    for record in ref_records:
        chromeName.append(record.description.split()[-1])
    for i in range(0, len(STR_data)):
        #print(STR_data['Chromosome'][i])
        if STR_data['Chromosome'][i] in chromeName:
            chromIndex = chromeName.index(STR_data['Chromosome'][i])
            ref_begin = sum(ref_len_list[:chromIndex])
        else:
            continue
        begin = ref_begin + int(STR_data['Start'][i])
        end = ref_begin +int(STR_data['Stop'][i])
        for j in range(int(begin / length), int((end / length))):
            bz[j] = 1
        # at avilible postion generate random beginning of seq
        if length > int(STR_data['Repeat Length'][i]):
            random_num = random.randint(0, length-int(STR_data['Repeat Length'][i]))
        else:
            random_num = 0
        b = begin - random_num
        final_data.append(ref_seq[b:(b + length)])
        final_description.append(STR_data['Actual Repeat'][i])
        num_positive += 1
    print('The number of positive items is: ', num_positive)

    # generate negative data
    num_negative = 0
    for i in range(0, int(len(ref_seq) / length)):
        if num_negative >= num_positive:
            break
        elif bz[i] == 1:
            continue
        else:
            final_data.append(ref_seq[i * length:(i * length + length)])
            final_description.append("NNNN")
            num_negative += 1
    print('The number of negative items is: ', num_negative)

    # generate label
    for i in range(num_positive):
        final_label.append(1)
    for i in range(num_negative):
        final_label.append(0)

    final_data = np.array(final_data)
    final_label = np.array(final_label)

    return final_data, final_label, final_description

def check(data):
    my_seq = Seq(data[0])
    read = SeqRecord(my_seq)
    read.id = str(data[1])
    read.description = data[2]
    return read

def saveDataAndLabel(data,label,description):
    data_zip = list(zip(data, label, description))
    items = list(range(len(label)))
    num_items = len(label)
    random.shuffle(items)

    train = items[0:int(0.8*num_items)]
    # valid = items[int(0.7*num_items):int(0.9*num_items)]
    test = items[int(0.8*num_items):num_items]
 
    train_record = []
    test_record = []

    for i in train:
        read = check(data_zip[i])
        train_record.append(read)
    for i in test:
        read = check(data_zip[i])  
        test_record.append(read)

    #save the data
    SeqIO.write(train_record, "../data/generate_STR/Escherichia_coli_train.fasta", "fasta")
    SeqIO.write(test_record, "../data/generate_STR/Escherichia_coli_test.fasta", "fasta")


if __name__ == '__main__':
    #This is using reference seq and STR csv description to generate positive and negetive STR seqences
    seq_length = 128
    data, label, description = get_data(seq_length)
    #save the data as fasta format
    saveDataAndLabel(data,label,description)