import numpy as n

def construct_confusion_matrix(class_lst,data):
    #print '--confusion matrix--'
    #init confusion matrix = Predicted x actual
    matrix = n.zeros((len(class_lst), len(class_lst)),dtype=n.int);
    for key in data:
        p = class_lst.index(data[key][0]);
        a = class_lst.index(data[key][1]);
        matrix[p,a]+=1
    displayStr = "\t"+"\t".join(class_lst) +"\n"
    for i_p in range(len(class_lst)):
        displayStr+=class_lst[i_p]+"\t"
        for j_a in range(len(class_lst)):
            displayStr+= str(matrix[i_p,j_a]) + "\t"
        displayStr+="\n"
    return displayStr

#test data
class_lst = ['a','b','c','d']
#'<data>':('<predicted>','actual')
data = {'d1':('a','b'),'d2':('b','c'),'d3':('c','d'),'d4':('a','d'),'d5':('b','b'),'d6':('b','b'),"d7":('a','d')}
