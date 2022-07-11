import sklearn
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy
import random
import sys

L=[]
LL=[]
#All training data
D_train={'summary':[], 'overall':[], 'score':[]}
#All test data
D_test={'summary':[], 'overall':[], 'score':[]}
# test data that we're going to predict
predict=D_test['summary']
#test data that we're going to compare to
Actual= D_test['score'] 
SVC = svm.SVC()
ms=pd.read_excel("Elec.xlsx",engine='openpyxl',header=None,names=None,dtype=None)
for row in ms.iterrows():
    L.append(row[1][0])
#L contains strings so let's transform them to dictionaries so we can access elements.
for j in range(len(L)): 
	LL.append(json.loads((L[j][0:len(L[j])])))
#train test split
train, test = train_test_split(LL, test_size=0.2,random_state=42)
for j in range(len(train)):
	D_train['overall'].append(train[j]['overall'])
	D_train['summary'].append(train[j]['summary'])
for j in range(len(test)):
	D_test['overall'].append(test[j]['overall'])
	D_test['summary'].append(test[j]['summary'])
#Define classes
def score(D): 
	for j in range(len(D['overall'])):
		if D['overall'][j]> 3:
			score='Positif'
			D['score'].append(score)
		elif D['overall'][j]< 3:
			score='Negatif'
			D['score'].append(score)
		elif D['overall'][j]==3: 
			score='Neutral'
			D['score'].append(score)
s=score(D_train)
ss=score(D_test)
Dneg_train={'summary':[], 'overall':[], 'score':[]}
Dpos_train={'summary':[], 'overall':[], 'score':[]}
Dneg_test={'summary':[], 'overall':[], 'score':[]}
Dpos_test={'summary':[], 'overall':[], 'score':[]}
#Resample to get rid rid of the 'Positif' bias since there are more positives than neg in the ds
def resample(D_train,Dpos_train,Dneg_train):
	for j in range(len(D_train['summary'])): 
		if D_train['score'][j]=='Negatif':
			Dneg_train['summary'].append(D_train['summary'][j]) 
			Dneg_train['overall'].append(D_train['overall'][j]) 
			Dneg_train['score'].append(D_train['score'][j])
	for j in range(len(D_train['summary'])):
		if D_train['score'][j]=='Positif':
			if len(Dpos_train['summary'])==len(Dneg_train['summary']): #fix this in case len is too small
				break
			else:
				Dpos_train['summary'].append(D_train['summary'][j]) 
				Dpos_train['overall'].append(D_train['overall'][j]) 
				Dpos_train['score'].append(D_train['score'][j])

	return Dneg_train,Dpos_train
resample(D_train,Dpos_train,Dneg_train)
resample(D_test,Dpos_test,Dneg_test)
D_train['summary'] = Dpos_train['summary'] + Dneg_train['summary']
D_train['overall'] = Dpos_train['overall'] + Dneg_train['overall']
D_train['score'] = Dpos_train['score'] + Dneg_train['score']
D_test['summary'] = Dpos_test['summary'] + Dneg_test['summary']
D_test['overall'] = Dpos_test['overall'] + Dneg_test['overall']
D_test['score'] = Dpos_test['score'] + Dneg_test['score']
#We can also use RandomState() and .seed for below
z_train= list(zip(D_train['summary'],D_train['overall'],D_train['score']))
z_test= list(zip(D_test['summary'],D_test['overall'],D_test['score']))
#Shuffle
random.shuffle(z_train)
random.shuffle(z_test)
#Re-assign, final dict 
D_train['summary'],D_train['overall'],D_train['score'] =zip(*z_train)
D_test['summary'],D_test['overall'],D_test['score'] =zip(*z_test)
#Tried with both Tfidf and count and Tfidf is better.
vectorizer = TfidfVectorizer(binary=True)
X_train = vectorizer.fit_transform(D_train['summary'])
#only transform
X_test=vectorizer.transform(D_test['summary'])
X_train=X_train.toarray()
X_test=X_test.toarray()
clf_SVM = svm.SVC(C=1, kernel='rbf')
model = clf_SVM.fit(X_train, D_train['score'])
Y_SVM = clf_SVM.predict(X_test)
from sklearn.metrics import accuracy_score
acc_SVM=accuracy_score(D_test['score'],Y_SVM)
from sklearn.metrics import f1_score
F1_SVM=f1_score(D_test['score'],Y_SVM,average=None)
#Decision Tree
from sklearn import tree
clf_DT = tree.DecisionTreeClassifier()
clf_DT = clf_DT.fit(X_train, D_train['score'])
Y_DT=clf_DT.predict(X_test)
acc_DT=accuracy_score(D_test['score'],Y_DT)
F1_DT=f1_score(D_test['score'],Y_DT,average=None)
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB=clf_NB.fit(X_train, D_train['score'])
Y_NB=clf_NB.predict(X_test)
acc_NB=accuracy_score(D_test['score'],Y_NB)
F1_NB=f1_score(D_test['score'],Y_NB,average=None)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(random_state=0)
clf_LR =clf_LR.fit(X_train, D_train['score'])
Y_LR=clf_LR.predict(X_test)
acc_LR=accuracy_score(D_test['score'],Y_LR)
F1_LR=f1_score(D_test['score'],Y_LR,average=None)
#Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_DA = LinearDiscriminantAnalysis()
clf_DA=clf_DA.fit(X_train, D_train['score'])
Y_DA=clf_DA.predict(X_test)
acc_DA=accuracy_score(D_test['score'],Y_DA)
F1_DA=f1_score(D_test['score'],Y_DA,average=None)
#Parameter tuning: SVM example: 
print(F1_NB)
print(F1_SVM)
print(F1_DA)
print(F1_DT)
print(F1_LR)
print(F1_NB)
#ROC curvz: first convert to int
D_test['score']= list(map(lambda x: 1 if x=='Positif' else 0, D_test['score']))
Y_SVM=list(map(lambda x: 1 if x=='Positif' else 0, Y_SVM))
Y_DT=list(map(lambda x: 1 if x=='Positif' else 0, Y_DT))
Y_NB=list(map(lambda x: 1 if x=='Positif' else 0, Y_NB))
Y_LR=list(map(lambda x: 1 if x=='Positif' else 0, Y_LR))
Y_DA=list(map(lambda x: 1 if x=='Positif' else 0, Y_DA))
fpr_SVM, tpr_SVM,_= metrics.roc_curve(D_test['score'],Y_SVM,pos_label=1)
fpr_DT, tpr_DT,_= metrics.roc_curve(D_test['score'],Y_DT,pos_label=1)
fpr_NB, tpr_NB,_= metrics.roc_curve(D_test['score'],Y_NB,pos_label=1)
fpr_LR, tpr_LR,_= metrics.roc_curve(D_test['score'],Y_LR,pos_label=1)
fpr_DA, tpr_DA,_= metrics.roc_curve(D_test['score'],Y_DA,pos_label=1)
plt.plot(fpr_DT,tpr_DT,label='Decision Trees')
plt.plot(fpr_NB,tpr_NB,label='Naive Bayes')
plt.plot(fpr_LR,tpr_LR,label='Logistic Regression')
plt.plot(fpr_DA,tpr_DA,label='discriminant analysis')
plt.plot(fpr_SVM,tpr_SVM,label='SVM')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
#since LR shows better solution we might as well use it as an example
print(confusion_matrix(D_test['score'],Y_LR))

