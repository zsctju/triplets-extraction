# -*- encoding:utf-8 -*-
__author__ = 'Suncong Zheng'
import cPickle
import os.path
import numpy as np
from PrecessEEdata import get_data_e2e
from Evaluate import evaluavtion_triple
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import  TimeDistributedDense, Dropout, Activation,Merge
from decodelayer import ReverseLayer2,LSTMDecoder_tag

def get_training_batch_xy_bias(inputsX, inputsY, max_s, max_t,
                          batchsize, vocabsize, target_idex_word,lossnum,shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        for idx, s in enumerate(excerpt):
            x[idx,] = inputsX[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                wordstr=''
                if word!=0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        yield x, y



def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)

def creat_binary_tag_LSTM( sourcevocabsize,targetvocabsize, source_W,input_seq_lenth ,output_seq_lenth ,
    hidden_dim ,emd_dim,loss='categorical_crossentropy',optimizer = 'rmsprop'):
    encoder_a = Sequential()
    encoder_b = Sequential()
    encoder_c = Sequential()
    l_A_embedding = Embedding(input_dim=sourcevocabsize+1,
                        output_dim=emd_dim,
                        input_length=input_seq_lenth,
                        mask_zero=True,
                        weights=[source_W])
    encoder_a.add(l_A_embedding)
    encoder_a.add(Dropout(0.3))
    encoder_b.add(l_A_embedding)
    encoder_b.add(Dropout(0.3))
    encoder_c.add(l_A_embedding)

    Model = Sequential()

    encoder_a.add(LSTM(hidden_dim,return_sequences=True))
    encoder_b.add(LSTM(hidden_dim,return_sequences=True,go_backwards=True))
    encoder_rb = Sequential()
    encoder_rb.add(ReverseLayer2(encoder_b))
    encoder_ab=Merge(( encoder_a,encoder_rb),mode='concat')
    Model.add(encoder_ab)

    decodelayer=LSTMDecoder_tag(hidden_dim=hidden_dim, output_dim=hidden_dim
                                         , input_length=input_seq_lenth,
                                        output_length=output_seq_lenth,
                                        state_input=False,
                                         return_sequences=True)
    Model.add(decodelayer)
    Model.add(TimeDistributedDense(targetvocabsize+1))
    Model.add(Activation('softmax'))
    Model.compile(loss=loss, optimizer=optimizer)
    return Model


def test_model(nn_model,testdata,index2word,resultfile=''):
    index2word[0]=''
    testx = np.asarray(testdata[0],dtype="int32")
    testy = np.asarray(testdata[1],dtype="int32")

    batch_size=50
    testlen = len(testx)
    testlinecount=0
    if len(testx)%batch_size ==0:
        testnum = len(testx)/batch_size
    else:
        extra_test_num = batch_size - len(testx)%batch_size
        extra_data = testx[:extra_test_num]
        testx=np.append(testx,extra_data,axis=0)
        extra_data = testy[:extra_test_num]
        testy=np.append(testy,extra_data,axis=0)
        testnum = len(testx)/batch_size

    testresult=[]
    for n in range(0,testnum):
        xbatch = testx[n*batch_size:(n+1)*batch_size]
        ybatch = testy[n*batch_size:(n+1)*batch_size]
        predictions = nn_model.predict(xbatch)

        for si in range(0,len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                ptag = []
                for word in sent:
                    next_index = np.argmax(word)
                    if next_index != 0:
                        next_token = index2word[next_index]
                        ptag.append(next_token)
                senty = ybatch[si]
                ttag=[]
                for word in senty:
                    next_token = index2word[word]
                    ttag.append(next_token)
                result = []
                result.append(ptag)
                result.append(ttag)
                testlinecount += 1
                testresult.append(result)
    cPickle.dump(testresult,open(resultfile,'wb'))
    P, R, F = evaluavtion_triple(testresult)
    print P, R, F
    return P, R, F


def train_e2e_model(eelstmfile, modelfile,resultdir,npochos,
                    lossnum=1,batch_size = 50,retrain=False):

    # load training data and test data
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k \
        = cPickle.load(open(eelstmfile, 'rb'))

    # train model
    x_train = np.asarray(traindata[0], dtype="int32")
    y_train = np.asarray(traindata[1], dtype="int32")

    nn_model = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                    source_W=source_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                    hidden_dim=k, emd_dim=k)
    if retrain:
        nn_model.load_weights(modelfile)
    #nn_model = CreatBinaryTagLSTM_Att(len(source_vob), len(target_vob), source_W, max_s, max_t, k, k)
    epoch = 0
    save_inter = 2
    saveepoch = save_inter
    maxF=0
    while (epoch < npochos):
        epoch = epoch + 1
        for x, y in get_training_batch_xy_bias(x_train, y_train, max_s, max_s,
                                          batch_size, len(target_vob),
                                            target_idex_word,lossnum,shuffle=True):
            nn_model.fit(x, y, batch_size=batch_size,
                         nb_epoch=1, show_accuracy=False, verbose=0)
            if epoch > saveepoch:
                saveepoch += save_inter
                resultfile = resultdir+"result-"+str(saveepoch)
                P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f\
                    = test_model(nn_model, testdata, target_idex_word,resultfile)
                if F > maxF:
                    maxF=F
                    save_model(nn_model, modelfile)
                print P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f
    return nn_model

def infer_e2e_model(eelstmfile, lstm_modelfile,resultfile):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    target_idex_word, max_s, k \
        = cPickle.load(open(eelstmfile, 'rb'))

    nnmodel = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob),targetvocabsize= len(target_vob),
                                    source_W=source_W,input_seq_lenth= max_s,output_seq_lenth= max_s,
                                    hidden_dim=k, emd_dim=k)

    nnmodel.load_weights(lstm_modelfile)
    P, R, F, pre1, rre1, fe1, pre2, rre2, fe2, tp1f, tp2f \
        = test_model(nnmodel, testdata, target_idex_word, resultfile)
    print P, R, F


if __name__=="__main__":

    alpha = 10
    maxlen = 50
    trainfile = "./data/demo/train_tag.json"
    testfile = "./data/demo/test_tag.json"
    w2v_file = "./data/demo/w2v.pkl"
    e2edatafile = "./data/demo/model/e2edata.pkl"
    modelfile = "./data/demo/model/e2e_lstmb_model.pkl"
    resultdir = "./data/demo/result/"

    retrain = True
    valid = False
    if not os.path.exists(e2edatafile):
        print "Precess lstm data...."
        get_data_e2e(trainfile,testfile,w2v_file,e2edatafile,maxlen=maxlen)
    if not os.path.exists(modelfile):
        print "Lstm data has extisted: "+e2edatafile
        print "Training EE model...."
        train_e2e_model(e2edatafile, modelfile,resultdir,
                     npochos=100,lossnum=alpha,retrain=False)
    else:
        if retrain:
            print "ReTraining EE model...."
            train_e2e_model(e2edatafile, modelfile, resultdir,
                         npochos=100,lossnum=alpha,retrain=retrain)



