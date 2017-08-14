
def count_predict_right_num(ptag,ttag):
    rmpair={}
    masktag = np.zeros(len(ptag))
    for i in range(0,len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist=[]
                e1=[]
                e2=[]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist=rmpair[type_e[0]]
                e1=eelist[0]
                e2=eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                ( ptag[j].__contains__("I") or  ptag[j].__contains__("L")):
                                j+=1
                            else:
                                break
                        e2.append((i, j))
                eelist[0]=e1
                eelist[1]=e2
                rmpair[type_e[0]] = eelist

    rightnum=0
    rightnume1 = 0
    rightnume2 = 0
    predictnum=0
    predictnume1 = 0
    predictnume2 = 0
    doublecount=0
    for type in rmpair:
        eelist = rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        if len(e1)>1 and len(e2)>1:
            doublecount+=min(len(e1),len(e2))
        if len(e1)>0 and len(e2)==0:
            predictnume1 = 1
        if len(e2) > 0 and len(e1) == 0:
            predictnume2 = 1
        for i in range(0,min(len(e1),len(e2))):
            predictnum+=1
            truemark=1
            truemarke1 = 1
            truemarke2 = 1

            for j in range(e1[i][0],e1[i][1]):
                if e1[i][0]>0 and ttag[e1[i][0]-1].__contains__(type):
                    truemark = 0
                    truemarke1 = 0
                    break
                if e1[i][1]<len(ttag) and ttag[e1[i][1]].__contains__(type):
                    truemark = 0
                    truemarke1 = 0
                    break
                if not ttag[j].__contains__(type) or not ttag[j].__contains__("1"):
                    truemark = 0
                    truemarke1 = 0
                    break
            for j in range(e2[i][0],e2[i][1]):
                if e2[i][0]>0 and ttag[e2[i][0]-1].__contains__(type):
                    truemark = 0
                    truemarke2 = 0
                    break
                if e2[i][1]<len(ttag) and ttag[e2[i][1]].__contains__(type):
                    truemark = 0
                    truemarke2 = 0
                    break
                if not ttag[j].__contains__(type) or not ttag[j].__contains__("2"):
                    truemark = 0
                    truemarke2 = 0
                    break
            if truemark ==1:
                rightnum+=1
            if truemarke1 ==1:
                rightnume1+=1
            if truemarke2 ==1:
                rightnume2+=1
    #print rightnum,predictnum
    return rightnum,predictnum,doublecount,rightnume1,rightnume2,predictnume1,predictnume2

