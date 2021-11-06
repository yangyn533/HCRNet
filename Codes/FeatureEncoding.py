import numpy as np
import collections
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index   


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index    


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]        
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        n=n//base
        ch3=chars[n%base]          
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  


def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100
    return vectors

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T') 
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array

def processFastaFile(seq):   
    phys_dic = {
    'A': [1,1,1],
    'U': [0,0,1],
    'C': [0,1,0],
    'G': [1,0,0]} 
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1
    return sequence_vector
    
def dpcp(seq):
    phys_dic = {
    #Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
    'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
    'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
    'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24,  0.14,],
    'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14,  -7.6, -19.2, -2.08, 0.08],
    'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
    'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
    'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
    'UG': [-0.01, -1.78, 3.32,  0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26,  0.17],
    'CA': [ 0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
    'CU': [-0.04, -1.5, 3.3,  0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
    'CC': [ -0.01, -1.78, 3.32, 0.3,  8.7, 32, -11.1, -12.2, -29.7, -3.26,  0.49],
    'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
    'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
    'GU': [-0.08, -1.27, 3.18, -0.8,  7,  31, -13.7, -6.6, -18.4, -0.93, 0.44],
    'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
    'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34 ]}
    
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2
    for i in range(0, seqLength-1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i+k]]
    return sequence_vector
        

def nd(seq, seq_length):
    seq = seq.strip()
    nd_list = [None] * seq_length
    for j in range(seq_length):
        #print(seq[0:j])
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j+1].count('A') / (j + 1), 3)
        elif seq[j] == 'U':
            nd_list[j] = round(seq[0:j+1].count('U') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j+1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j+1].count('G') / (j + 1), 3)     
    return np.array(nd_list)

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
    

def dealwithdata(protein):
    seq_length = 101
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    tris4 = get_4_trids()
    dataX = []
    dataY = []
    with open('~./datapath/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                probMatr = processFastaFile(line)
                probMatr_ND = nd(line,seq_length)
                probMatr_NDCP = np.column_stack((probMatr,probMatr_ND))
                probMatr_DPCP = dpcp(line)/101
                probMatr_NDPCP = np.column_stack((probMatr_NDCP,probMatr_DPCP))                
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                Feature_Encoding = np.column_stack((probMatr_NDPCP,Kmer))  
                dataX.append(Feature_Encoding.tolist())
    with open('~./datapath/' + protein + '/negative') as f:          
        for line in f:
            if '>' not in line:
                line = line.replace('T', 'U').strip()
                probMatr = processFastaFile(line)
                probMatr_ND = nd(line,seq_length)
                probMatr_NDCP = np.column_stack((probMatr,probMatr_ND))
                probMatr_DPCP = dpcp(line)/101
                probMatr_NDPCP = np.column_stack((probMatr_NDCP,probMatr_DPCP))                 
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                Feature_Encoding = np.column_stack((probMatr_NDPCP,Kmer))
                dataX.append(Feature_Encoding.tolist())
    dataX = np.array(dataX)
    return dataX





    
 
