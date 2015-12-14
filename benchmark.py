
import numpy as np
from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


from sklearn.svm import SVC

def get_x_y():
    df1 = pd.read_csv('train_x.csv')
    df2 = pd.read_csv('train_y.csv')
    x_col = df1.columns[1:]
    return df1[x_col], df2.y
def get_train_test_sets(x,y):
    '''
    x_train, x_test, y_train, y_test
    '''

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3, random_state=0)
    return x_train, x_test, y_train, y_test 


def svm_score(x,y):
    ''' 
    cv = 2 , Accuracy: 0.90 (+/- 0.00)
    cv = 4 , Accuracy: 0.90 (+/- 0.00)
    '''

    clf = SVC()
    scores = cross_val_score(clf,x,y, cv=4,n_jobs=-1)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)



def logisticre_score(x,y):
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=1.0, penalty='l2')
    lg.fit(x,y)
    scores = cross_val_score(lg,x,y, cv=5,n_jobs=-1)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def plot_ligistic_roc():
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=1.0, penalty='l2')
    x,y = get_x_y()
    x_train, x_test, y_train, y_test = get_train_test_sets(x,y)
    lg.fit(x_train, y_train)
    r = lg.predict_proba(x_test) 
    y0_score, y1_score = r[:,0], r[:,1]
    plot_roc(y_test, y1_score)

def get_auc(y, y_score):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y, y_score)
    
    return auc(fpr, tpr)

def get_cross_auc(x,y):
    '''
    c=1.0, penalty=l2, 0.50193326693284135
    '''
    from sklearn.cross_validation import KFold
    kf = KFold(n=len(x), n_folds=5, shuffle=False)
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=1.0, penalty='l1')
    auc_list = []
    for train_index, test_index in kf:
        x_train = x.ix[train_index]
        y_train = y.ix[train_index]
        x_test = x.ix[test_index]
        y_test = y.ix[test_index]
        lg.fit(x_train, y_train)
        r = lg.predict_proba(x_test)
        fpr, tpr, _ =  roc_curve(y_test, r[:,1], 1)
        auc_list.append(auc(fpr, tpr))

    return sum(auc_list)/len(auc_list)


def plot_roc(y, y_score, pos_label=None):
    fpr, tpr, _ = roc_curve(y, y_score, pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)'%roc_auc)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating charateristic example')
    plt.legend(loc='lower right')
    plt.show()

    
if __name__ == '__main__':
    x,y = get_x_y()
    #svm_score(x,y)
    #logisticre_score(x,y)















