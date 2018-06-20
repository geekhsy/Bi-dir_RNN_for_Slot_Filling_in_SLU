# coding=utf-8
import random
import numpy as np
import cPickle

flatten = lambda l: [item for sublist in l for item in sublist]  # Two-dimensional development into one dimension
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

def atisfold(fold):
    assert fold in range(5)
    f = open('AtisData/atis.fold'+str(fold)+'.pkl')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts

def get_data(fold):
    train, valid, test, dic = atisfold(fold) 

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w  = dict((v,k) for k,v in w2idx.iteritems())
    idx2la = dict((v,k) for k,v in labels2idx.iteritems())

    
    test_x,  test_ne,  test_label  = test
    train_x, train_ne, train_label = train

    train_set=[]
    for e in ['train']:
        for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
            Wsent="BOS "
            Tsent=" "
            for wx, la in zip(sw, sl):
                Wsent=Wsent+idx2w[wx]+" "
                Tsent=Tsent+idx2la[la]+" "
            Fsent=Wsent+"EOS\t "+Tsent+"\n"
            train_set.append(Fsent)

    test_set=[]
    for e in ['test']:
        for sw, se, sl in zip(eval(e+'_x'), eval(e+'_ne'), eval(e+'_label')):
            Wsent="BOS "
            Tsent=" "
            for wx, la in zip(sw, sl):
                Wsent=Wsent+idx2w[wx]+" "
                Tsent=Tsent+idx2la[la]+" "
            Fsent=Wsent+"EOS\t "+Tsent+"\n"
            test_set.append(Fsent)

    return train_set,test_set
    

def data_pipeline(data, length=50):
    data = [t[:-1] for t in data]  # Remove'\n'
    # According to the line like this：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # Segmented into such [original sentence words, labeled sequence，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # Remove BOS and EOS, and remove the corresponding label in the corresponding label sequence
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []
    # padding，The end of the original sequence and label sequence +<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)
        data = list(zip(sin, sout, intent))
    return data


def get_info_from_training_data(data):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    # Generation of word2index
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    # Generation of index2word
    # index2word = {v: k for k, v in word2index.items()}

    # Generation of tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # Generation of index2tag
    # index2tag = {v: k for k, v in tag2index.items()}

    # Generation of intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    # Generation of index2intent
    # index2intent = {v: k for k, v in intent2index.items()}
    return word2index, tag2index,  intent2index


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch


def to_index(train, word2index, slot2index, intent2index):
    new_train = []
    for sin, sout, intent in train:
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
    return new_train