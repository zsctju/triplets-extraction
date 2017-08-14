#coding=utf-8
__author__ = 'Suncong Zheng'
import json
import unicodedata
import nltk

def Static(source_train_json,isTrain=True):
    file = open(source_train_json, 'r')
    sentences_0 = file.readlines()
    sentences=[]
    for s in sentences_0:
        if not sentences.__contains__(s):
            sentences.append(s)
    rmlabelmap = {}
    elendis = {}
    for line in sentences:

        sent = json.loads(line.strip('\r\n'))
        sentText = str(unicodedata.normalize('NFKD', sent['sentText']).encode('ascii','ignore')).rstrip('\n').rstrip('\r')
        try:
            tags=[]
            tokens = nltk.word_tokenize(sentText)
            for i in range(0,len(tokens)):
                tags.append('O')
            #tokens = sentText.split()
            relationMentions = []
            entityMentions = []
            emStartIndexes = set()
            emIndexByText = {}
            for em in sent['entityMentions']:
                emText = unicodedata.normalize('NFKD', em['text']).encode('ascii','ignore')
                if emText not in emIndexByText:
                    start, end = find_index(tokens, emText.split())
                else:
                    offset = emIndexByText[emText][-1][1]
                    start, end = find_index(tokens[offset:], emText.split())
                    start += offset
                    end += offset
                if start != -1 and end != -1:
                    if end <= start:
                        continue
                    emStartIndexes.add(start)
                    if emText not in emIndexByText:
                        emIndexByText[emText] = [(start, end)]
                    else:
                        emIndexByText[emText].append((start, end))
                    entityMentions.append({'start':start, 'end':end, 'labels':em['label'].split(',')})
            emStartIndexes = sorted(list(emStartIndexes))
            orderByStartIdxMap = {}
            for i in range(len(emStartIndexes)):
                orderByStartIdxMap[emStartIndexes[i]] = i
            visitedEmPairs = {}
            numOfEMBetweenMap = {}
            for rm in sent['relationMentions']:
                try:
                    start1 = -1
                    end1 = -1
                    start2 = -1
                    end2 = -1
                    em1 = unicodedata.normalize('NFKD', rm['em1Text']).encode('ascii','ignore')
                    em2 = unicodedata.normalize('NFKD', rm['em2Text']).encode('ascii','ignore')
                    if isTrain:
                        start1 = emIndexByText[em1][-1][0]
                        end1 = emIndexByText[em1][-1][1]
                        start2 = emIndexByText[em2][-1][0]
                        end2 = emIndexByText[em2][-1][1]
                    else:
                        for em1Index in emIndexByText[em1]:
                            flag = False
                            for em2Index in emIndexByText[em2]:
                                if (em1Index, em2Index) not in visitedEmPairs:
                                    start1 = em1Index[0]
                                    end1 = em1Index[1]
                                    start2 = em2Index[0]
                                    end2 = em2Index[1]
                                    flag = True
                                    break
                            if flag:
                                break
                    numOfEMBetween = 0
                    if start2 > start1:
                        numOfEMBetween = orderByStartIdxMap[start2] - orderByStartIdxMap[start1] - 1
                    elif start2 < start1:
                        numOfEMBetween = orderByStartIdxMap[start1] - orderByStartIdxMap[start2] - 1
                    if start1 != -1 and end1 != -1 and start2 != -1 and end2 != -1:
                        numOfEMBetweenMap[(start1, end1), (start2, end2)] = numOfEMBetween
                        if ((start1, end1), (start2, end2)) in visitedEmPairs:
                            visitedEmPairs[((start1, end1), (start2, end2))].add(rm['label'])
                        else:
                            visitedEmPairs[((start1, end1), (start2, end2))] = set([rm['label']])
                except Exception as e:
                    a=1
                    #print 'index error: ', e.message, e.args
                    #print sent['articleId'], sent['sentId'], ' : ', rm
            if len(visitedEmPairs) > 0:

                for emPair in visitedEmPairs:
                    valid = True
                    for ei in range(emPair[0][0], emPair[0][1]):
                        if not tags[ei].__eq__('O'):
                            valid = False
                            break
                    for ei in range(emPair[1][0], emPair[1][1]):
                        if not tags[ei].__eq__('O'):
                            valid = False
                            break
                    if valid and NoOverlap(emPair[0][0],emPair[0][1],emPair[1][0],emPair[1][1]):
                        for rmlabel in visitedEmPairs[emPair]:
                            if not rmlabel.__eq__("None"):
                                if not rmlabelmap.__contains__(rmlabel):
                                    rmlabelmap[rmlabel] = 1
                                else:
                                    rmlabelmap[rmlabel] += 1
                                lene = emPair[0][1] - emPair[0][0]
                                if elendis.__contains__(lene):
                                    elendis[len] = elendis[lene]+1
                                else:
                                    elendis[lene] =  1
                                lene = emPair[1][1] - emPair[1][0]
                                if elendis.__contains__(lene):
                                    elendis[lene] = elendis[lene] + 1
                                else:
                                    elendis[lene] = 1
                                break
        except Exception as e:
            a=1
            #print 'index error: ', e.message, e.args
    return rmlabelmap,elendis

def tag_sent(source_json,tag_json,isTrain=True):
    """
    Tagging the text based on the tagging schema
    :param the source_json file: the sent text, entity mentions, relation mentions et.al
    :return: the tag_json file: the sent text, the tag sequences
    """
    train_json_file = open(tag_json, 'w', 0)
    file = open(source_json, 'r')
    sentences_0 = file.readlines()
    sentences=[]
    for s in sentences_0:
        if not sentences.__contains__(s):
            sentences.append(s)
    AllRmcount = 0
    LableRmCount=0
    TagRmCount=0
    rmlabelmap = {}
    countdouble=0
    for line in sentences:
        doubelrelnum = []
        doublemark=0
        sent = json.loads(line.strip('\r\n'))
        sentText = str(unicodedata.normalize('NFKD', sent['sentText']).encode('ascii','ignore')).rstrip('\n').rstrip('\r')
        try:
            tags=[]
            tokens = nltk.word_tokenize(sentText)
            for i in range(0,len(tokens)):
                tags.append('O')
            #tokens = sentText.split()
            relationMentions = []
            entityMentions = []
            emStartIndexes = set()
            emIndexByText = {}
            for em in sent['entityMentions']:
                emText = unicodedata.normalize('NFKD', em['text']).encode('ascii','ignore')
                if emText not in emIndexByText:
                    start, end = find_index(tokens, emText.split())
                else:
                    offset = emIndexByText[emText][-1][1]
                    start, end = find_index(tokens[offset:], emText.split())
                    start += offset
                    end += offset
                if start != -1 and end != -1:
                    if end <= start:
                        continue
                    emStartIndexes.add(start)
                    if emText not in emIndexByText:
                        emIndexByText[emText] = [(start, end)]
                    else:
                        emIndexByText[emText].append((start, end))
                    entityMentions.append({'start':start, 'end':end, 'labels':em['label'].split(',')})
            emStartIndexes = sorted(list(emStartIndexes))
            orderByStartIdxMap = {}
            for i in range(len(emStartIndexes)):
                orderByStartIdxMap[emStartIndexes[i]] = i
            visitedEmPairs = {}
            numOfEMBetweenMap = {}
            for rm in sent['relationMentions']:
                if not rmlabelmap.__contains__(rm['label']):
                    rmlabelmap[rm['label']] = 1
                else:
                    rmlabelmap[rm['label']] += 1
                if not rm['label'].__eq__("None"):
                    LableRmCount+=1
                try:
                    start1 = -1
                    end1 = -1
                    start2 = -1
                    end2 = -1
                    em1 = unicodedata.normalize('NFKD', rm['em1Text']).encode('ascii','ignore')
                    em2 = unicodedata.normalize('NFKD', rm['em2Text']).encode('ascii','ignore')
                    if isTrain:
                        start1 = emIndexByText[em1][-1][0]
                        end1 = emIndexByText[em1][-1][1]
                        start2 = emIndexByText[em2][-1][0]
                        end2 = emIndexByText[em2][-1][1]
                    else:
                        for em1Index in emIndexByText[em1]:
                            flag = False
                            for em2Index in emIndexByText[em2]:
                                if (em1Index, em2Index) not in visitedEmPairs:
                                    start1 = em1Index[0]
                                    end1 = em1Index[1]
                                    start2 = em2Index[0]
                                    end2 = em2Index[1]
                                    flag = True
                                    break
                            if flag:
                                break
                    numOfEMBetween = 0
                    if start2 > start1:
                        numOfEMBetween = orderByStartIdxMap[start2] - orderByStartIdxMap[start1] - 1
                    elif start2 < start1:
                        numOfEMBetween = orderByStartIdxMap[start1] - orderByStartIdxMap[start2] - 1
                    if start1 != -1 and end1 != -1 and start2 != -1 and end2 != -1:
                        numOfEMBetweenMap[(start1, end1), (start2, end2)] = numOfEMBetween
                        if ((start1, end1), (start2, end2)) in visitedEmPairs:
                            visitedEmPairs[((start1, end1), (start2, end2))].add(rm['label'])
                        else:
                            visitedEmPairs[((start1, end1), (start2, end2))] = set([rm['label']])
                except Exception as e:
                    a=1
                    #print 'index error: ', e.message, e.args
                    #print sent['articleId'], sent['sentId'], ' : ', rm
            if len(visitedEmPairs) > 0:
                for emPair in visitedEmPairs:
                    AllRmcount += 1
                    valid = True
                    for ei in range(emPair[0][0], emPair[0][1]):
                        if not tags[ei].__eq__('O'):
                            valid = False
                            break
                    for ei in range(emPair[1][0], emPair[1][1]):
                        if not tags[ei].__eq__('O'):
                            valid = False
                            break
                    if valid and no_overlap(emPair[0][0],emPair[0][1],emPair[1][0],emPair[1][1]):
                        for rmlabel in visitedEmPairs[emPair]:
                            if not rmlabel.__eq__("None"):
                                if not doubelrelnum.__contains__(rmlabel):
                                    doubelrelnum.append(rmlabel)
                                else:
                                    countdouble+=1
                                    doublemark=1
                                TagRmCount+=1
                                if emPair[0][1] - emPair[0][0] == 1:
                                    tags[emPair[0][0]] = rmlabel+"__E1S"
                                elif emPair[0][1] - emPair[0][0] == 2:
                                    tags[emPair[0][0]] = rmlabel + "__E1B"
                                    tags[emPair[0][0]+1] = rmlabel + "__E1L"
                                else:
                                    tags[emPair[0][0]] = rmlabel + "__E1B"
                                    tags[emPair[0][1]-1] = rmlabel + "__E1L"
                                    for ei in range(emPair[0][0]+1,emPair[0][1]-1):
                                        tags[ei] = rmlabel + "__E1I"

                                if emPair[1][1] - emPair[1][0] == 1:
                                    tags[emPair[1][0]] = rmlabel+"__E2S"
                                elif emPair[1][1] - emPair[1][0] == 2:
                                    tags[emPair[1][0]] = rmlabel + "__E2B"
                                    tags[emPair[1][0]+1] = rmlabel + "__E2L"
                                else:
                                    tags[emPair[1][0]] = rmlabel + "__E2B"
                                    tags[emPair[1][1]-1] = rmlabel + "__E2L"
                                    for ei in range(emPair[1][0]+1,emPair[1][1]-1):
                                        tags[ei] = rmlabel + "__E2I"
                                break       # only one relation type for each word

            newsent = dict()
            newsent['tokens'] = tokens
            newsent['tags'] = tags
            #if doublemark==1:
            #    print tags
            train_json_file.write(json.dumps(newsent)+'\n')
        except Exception as e:
            print e.message, e.args
    #print LableRmCount,TagRmCount,len(rmlabelmap),AllRmcount,countdouble

def no_overlap(index11,index12,index21,index22):
    if index11>=index22:
        return True
    if index21>=index12:
        return True
    return False

def find_index(sen_split, word_split):
    index1 = -1
    index2 = -1
    for i in range(len(sen_split)):
        if str(sen_split[i]) == str(word_split[0]):
            flag = True
            k = i
            for j in range(len(word_split)):
                if word_split[j] != sen_split[k]:
                    flag = False
                if k < len(sen_split) - 1:
                    k+=1
            if flag:
                index1 = i
                index2 = i + len(word_split)
                break
    return index1, index2

if __name__ == "__main__":
    infile1 = "./data/demo/train.json"
    infile2 = "./data/demo/train_tag.json"
    infile3 = "./data/demo/test.json"
    infile4 = "./data/demo/test_tag.json"
    tag_sent(infile1,infile2,isTrain=True)
    tag_sent(infile3,infile4,isTrain=False)







