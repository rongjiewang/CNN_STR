import os
import pandas as pd
import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import math

convertIDTable = {'sacCer3':{'NC_001133.9':'chrI',
                            'NC_001134.8':'chrII',
                            'NC_001135.5':'chrIII',
                            'NC_001136.10':'chrIV',
                            'NC_001137.3':'chrV',
                            'NC_001138.5':'chrVI',
                            'NC_001139.9':'chrVII',
                            'NC_001140.6':'chrVIII',
                            'NC_001141.2':'chrIX',
                            'NC_001142.9':'chrX',
                            'NC_001143.9':'chrXI',
                            'NC_001144.5':'chrXII',
                            'NC_001145.3':'chrXIII',
                            'NC_001146.8':'chrXIV',
                            'NC_001147.6':'chrXV',
                            'NC_001148.4':'chrXVI',
                            'NC_001224.1':'chrM'},
                  'ASM522150v1':{'NZ_CP028592.1':'NZ_CP028592.1'},
                  'dm6':{   'NC_004354.4':'chrX',
                            'NT_033779.5':'chr2L',
                            'NT_033778.4':'chr2R',
                            'NT_037436.4':'chr3L',
                            'NT_033777.3':'chr3R',
                            'NC_004353.4':'chr4',
                            'NC_024512.1':'chrY'},
                  'hg38_chr1':{'NC_000001.11':'chr1'}}

# read reference sequence data
def read_fasta(curent_data_name):
    dataName = "./data/ref/" + curent_data_name + ".fasta"
    data_path = os.path.abspath(os.path.join(os.getcwd(), dataName))
    records = list(SeqIO.parse(data_path, "fasta"))
    seqence = ""
    seq_len_list = []
    record_name = []
    for record in records:
        if record.id in convertIDTable[curent_data_name].keys():
            seqence += str(record.seq)
            seq_len_list.append(len(record.seq))
            record_name.append(record.id)
    return record_name, seqence, seq_len_list

# read STR csv data
def get_STR_csv(curent_data_name):
    dataName = "./data/tsv/" + curent_data_name + ".tsv"
    test_path = os.path.abspath(os.path.join(os.getcwd(), dataName))
    data = pd.read_csv(test_path, sep='\t', header=None,
                       names=['Chromosome', 'Start', 'Stop', 'Repeat Class', 'Repeat Length', 'Strand', '# Units',
                              'Actual Repeat', 'Gene Name', 'Gene Start', 'Gene Stop', 'Gene Strand', 'Annotation',
                              'Distance from TSS'])
    max_len = data['Repeat Length'].max()
    print("max STR length: ", max_len)
    max_len = max([len(x) for x in data['Actual Repeat']])
    print("max Unit length: ",max_len)
    return data

def get_data(cut_length, curent_data):
    STR_data = get_STR_csv(curent_data)
    ref_records_name, ref_seq, ref_len_list = read_fasta(curent_data)
    # initialization
    bz = []
    final_data = []
    final_label = []
    final_description = []

    for i in range(0, math.ceil(len(ref_seq) / cut_length)):
        bz.append(0)
    # generate positive data
    num_positive = 0
    print("The number of STR is:", len(STR_data))
    for i in range(0, len(STR_data)):
        if STR_data['Chromosome'][i] in inverted_IDTable.keys():
            ref_name = inverted_IDTable[STR_data['Chromosome'][i]]
            chromIndex = ref_records_name.index(ref_name)
            ref_begin = sum(ref_len_list[:chromIndex])
        else:
            continue
        begin = ref_begin + int(STR_data['Start'][i])
        end = ref_begin +int(STR_data['Stop'][i])
        for j in range(math.floor(begin / cut_length), math.ceil((end / cut_length))):            
            bz[j] = 1

        # at avilible postion generate random beginning of seq
        if int(STR_data['Start'][i]) < cut_length:
            continue #escape the biginning sequences
        if cut_length > int(STR_data['Repeat Length'][i]):
            random_num = random.randint(0, (cut_length-int(STR_data['Repeat Length'][i])))
        else:
            random_num = 0
        b = begin - random_num
        final_data.append(ref_seq[b:(b + cut_length)])
        final_description.append((STR_data['Actual Repeat'][i], random_num, \
            len(STR_data['Actual Repeat'][i]), STR_data['# Units'][i]))
        num_positive += 1
    print('The number of positive items is: ', num_positive)

    # generate negative data
    num_negative = 0
    for i in range(0, int(len(ref_seq) / cut_length)):
        if num_negative >= num_positive:
            break
        elif bz[i] == 1:
            continue
        else:
            final_data.append(ref_seq[i * cut_length:(i * cut_length + cut_length)])
            final_description.append(("NNNN",0,0,0))
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
    my_seq = Seq(data[0].upper())
    read = SeqRecord(my_seq)
    read.id = str(data[1])
    read.description = ""
    if data[1]:
        read.description += "Actual Repeat:" + str(data[2][0]) + " "
        read.description += "Start:" + str(data[2][1]) + " "
        read.description += "Repeat Length:" + str(data[2][2]) + " "
        read.description += "# Units:" + str(data[2][3])
    return read

def saveDataAndLabel(data,label,description,curent_data):
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
    saveName_train = "./data/generate_STR/"+curent_data + "_train.fasta"
    saveName_test = "./data/generate_STR/"+curent_data + "_test.fasta"
    SeqIO.write(train_record, saveName_train, "fasta")
    SeqIO.write(test_record, saveName_test, "fasta")

if __name__ == '__main__':
    #This is using reference seq and STR csv description to generate positive and negetive STR seqences
    refNameList = ['ASM522150v1','sacCer3','dm6','hg38_chr1']
    for i in refNameList:
        print("curent dataset:",i)
        inverted_IDTable = {value: key for key, value in convertIDTable[i].items()}
        seq_length = 128
        data, label, description = get_data(seq_length,i)
        #save the data as fasta format
        saveDataAndLabel(data,label,description,i)
    print("finished")
