#coding=utf-8
__author__ = 'Suncong Zheng'
import numpy as np
import cPickle
import json
import re

def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = cPickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W

def make_idx_data_index_EE_LSTM(file,max_s,source_vob,target_vob):
    """
    Coding the word sequence and tag sequence based on the digital index which provided by source_vob and target_vob
    :param the tag file: word sequence and tag sequence
    :param the word index map and tag index map: source_vob,target_vob
    :param the maxsent lenth: max_s
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """

    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        t_sent = sent['tags']
        data_t = []
        data_s = []
        if len(s_sent) > max_s:
            i=max_s-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        else:
            num=max_s-len(s_sent)
            for inum in range(0,num):
                data_s.append(0)
            i=len(s_sent)-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        data_s_all.append(data_s)
        if len(t_sent) > max_s:
            for i in range(0,max_s):
                data_t.append(target_vob[t_sent[i]])
        else:
            for word in t_sent:
                data_t.append(target_vob[word])
            while len(data_t)< max_s:
                data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all,data_t_all]

def get_word_index(train,test):
    """
    Give each word an index
    :param the train file and the test file
    :return: the word_index map, the index_word map, the tag_index map, the index_tag map,
    the max lenth of word sentence
    """
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount=1
    max_s=0
    max_t=0
    f = open(train,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()

        target = sent['tags']

        if target.__len__()> max_t:
            max_t = target.__len__()
        for word in target:
            if not target_vob.__contains__(word):
                target_vob[word] = tarcount
                target_idex_word[tarcount] = word
                tarcount += 1
    f.close()

    f = open(test,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()

        target = sent['tags']
        if not source_vob.__contains__(target[0]):
                source_vob[target[0]] = count
                sourc_idex_word[count] = target[0]
                count += 1
        if target.__len__()> max_t:
            max_t = target.__len__()
        for word in target:
            if not target_vob.__contains__(word):
                target_vob[word] = tarcount
                target_idex_word[tarcount] = word
                tarcount += 1
    f.close()
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count+=1
    if not source_vob.__contains__("UNK"):
        source_vob["UNK"] = count
        sourc_idex_word[count] = "UNK"
        count+=1
    return source_vob,sourc_idex_word,target_vob,target_idex_word,max_s

def get_data_e2e(trainfile,testfile,w2v_file,eelstmfile,maxlen = 50):
    """
    Converts the input files  into the end2end model input formats
    :param the train tag file: produced by TaggingScheme.py
    :param the test tag file: produced by TaggingScheme.py
    :param the word2vec file: Extracted form the word2vec resource
    :param: the maximum sentence length we want to set
    :return: tthe end2end model formats data: eelstmfile
    """
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = \
    get_word_index(trainfile, testfile)

    print "source vocab size: " + str(len(source_vob))
    print "target vocab size: " + str(len(target_vob))

    source_w2v ,k ,source_W= load_vec_pkl(w2v_file,source_vob)

    print "word2vec loaded!"
    print "num words in source word2vec: " + str(len(source_w2v))+\
          "source  unknown words: "+str(len(source_vob)-len(source_w2v))

    if max_s > maxlen:
        max_s = maxlen

    print 'max soure sent lenth is ' + str(max_s)

    train = make_idx_data_index_EE_LSTM(trainfile,max_s,source_vob,target_vob)
    test = make_idx_data_index_EE_LSTM(testfile, max_s, source_vob, target_vob)

    print "dataset created!"
    cPickle.dump([train,test,source_W,source_vob,sourc_idex_word,
                  target_vob,target_idex_word,max_s,k],open(eelstmfile,'wb'))

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def peplacedigital(s):
    if len(s)==1:
        s='1'
    elif len(s)==2:
        s='10'
    elif len(s)==3:
        s='100'
    else:
        s='1000'
    return s
