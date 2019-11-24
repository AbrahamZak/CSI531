'''
Classifier Evaluation

@author: Abraham Zakharov
'''

import numpy as np
from sklearn import tree 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def SVM (X,Y):
    #c values
    C = [0.01, 0.1, 1, 10, 100]
    #Results
    Results = []
    #loop through c values
    for vals in C:
        clf = svm.SVC(kernel='linear', C=vals)
        Results.append(cross_val_score(clf, X, Y, cv=10))
        
    ResultsAvg = []
    #get the averages for each C
    for s in Results:
        ResultsAvg.append(sum(s) / len(s))
    
    #print(ResultsAvg)
    
    #plot the data
    plt.plot(C, ResultsAvg)
    plt.ylabel('Average F measure')
    plt.xlabel('C')
    plt.title('Average F measure of C values in linear SVM ')
    plt.show()
        
def trees(X,Y):
    #max leaf nodes
    k = [2, 5, 10, 20]
    #IG and gini cross val scores
    DTig = []
    DTgini = []
    
    #loop through max leaf nodes
    for nodes in k:
        #create the tree
        clf = tree.DecisionTreeClassifier(criterion= 'entropy', max_leaf_nodes=nodes)
        #store cross val score
        DTig.append(cross_val_score(clf, X, Y, cv=10))
        
    #same process for gini
    for nodes in k:
        #create the tree
        clf = tree.DecisionTreeClassifier(criterion= 'gini', max_leaf_nodes=nodes)
        #store cross val score
        DTgini.append(cross_val_score(clf, X, Y, cv=10))
    
    DTigAvg = []
    DTginiAvg = []
    #get the averages for each k in IG
    for s in DTig:
        DTigAvg.append(sum(s) / len(s))
        
    #get the averages for each k in GINI
    for s in DTgini:
        DTginiAvg.append(sum(s) / len(s))
        
    #print(DTigAvg)
    #print(DTginiAvg)
    #plot the data
    plt.plot(k, DTigAvg, label = 'Information Gain')
    plt.plot(k, DTginiAvg, label = 'GINI')
    plt.ylabel('Average F measure')
    plt.xlabel('Leaf Nodes (k)')
    plt.title('Decision Trees average F-measure in 10-fold cross validation as a function of k')
    plt.legend()
    plt.show()
    
    
def compare(X,Y,Xtest,y_true):
    #best SVM from earlier results
    SVMclass = svm.SVC(kernel='linear', C=0.01).fit(X, Y)
    #best IG from earlier results
    IGclass = tree.DecisionTreeClassifier(criterion= 'gini', max_leaf_nodes=5).fit(X, Y)
    #best GINI from earlier results
    GINIclass = tree.DecisionTreeClassifier(criterion= 'gini', max_leaf_nodes=10).fit(X, Y)
    #LDA
    LDAclass = LinearDiscriminantAnalysis().fit(X, Y)  
    
    #predict for all classes
    SVM_y_pred = SVMclass.predict(Xtest)
    IG_y_pred = IGclass.predict(Xtest)
    GINI_y_pred = GINIclass.predict(Xtest)
    LDA_y_pred = LDAclass.predict(Xtest)

    #average class precision 
    SVM_precision = average_precision_score(y_true.astype(np.float), SVM_y_pred.astype(np.float))
    IG_precision = average_precision_score(y_true.astype(np.float), IG_y_pred.astype(np.float))
    GINI_precision = average_precision_score(y_true.astype(np.float), GINI_y_pred.astype(np.float))
    LDA_precision = average_precision_score(y_true.astype(np.float), LDA_y_pred.astype(np.float))
    
    #average class recall  
    SVM_rec = recall_score(Ytest, SVM_y_pred, average='weighted') 
    IG_rec = recall_score(Ytest, IG_y_pred, average='weighted') 
    GINI_rec = recall_score(Ytest, GINI_y_pred, average='weighted') 
    LDA_rec = recall_score(Ytest, LDA_y_pred, average='weighted') 
    
    #average class F-measure
    SVM_f = f1_score(Ytest, SVM_y_pred, average='weighted')
    IG_f = f1_score(Ytest, IG_y_pred, average='weighted')
    GINI_f = f1_score(Ytest, GINI_y_pred, average='weighted')
    LDA_f = f1_score(Ytest, LDA_y_pred, average='weighted')
    
    labels = ['Precision', 'Recall', 'F-Measure']
    SVM_measures = [SVM_precision, SVM_rec, SVM_f]
    IG_measures = [IG_precision, IG_rec, IG_f]
    GINI_measures = [GINI_precision, GINI_rec, GINI_f]
    LDA_measures = [LDA_precision, LDA_rec, LDA_f]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/4, SVM_measures, width/4, label='SVM')
    ax.bar(x + width/4, IG_measures, width/4, label='IG')
    ax.bar(x, GINI_measures, width/4, label='GINI')
    ax.bar(x + width/2, LDA_measures, width/4, label='LDA')
    
    ax.set_ylabel('Values')
    ax.set_title('Classifier Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    fig.tight_layout()
    plt.show()
    
    #print(SVM_measures)
    #print(IG_measures)
    #print(GINI_measures)
    #print(LDA_measures)
    
if __name__ == '__main__':
    #split the data
    X = np.genfromtxt('cancer-data-train.csv', delimiter=",", usecols=range(0,30))
    Y = np.genfromtxt('cancer-data-train.csv', delimiter=",", usecols=(30), dtype=str)
    Y[Y == 'M'] = 1
    Y[Y == 'B'] = 0
    #SVM
    SVM(X,Y)
    #trees
    trees(X,Y)
    Xtest = np.genfromtxt('cancer-data-test.csv', delimiter=",", usecols=range(0,30))
    Ytest = np.genfromtxt('cancer-data-test.csv', delimiter=",", usecols=(30), dtype=str)
    Ytest[Ytest == 'M'] = 1
    Ytest[Ytest == 'B'] = 0
    #comparison
    compare(X,Y,Xtest,Ytest)
    pass

