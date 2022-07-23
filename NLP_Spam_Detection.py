'''
This project is about applying NLP for detecting spam email from non-spam (ham),  
which is one of the trendiest topics in the online social world and it is  
critically important in terms of stopping fraud and trash emails

I utilized the Bag of Words (BOW) technique in order to transform our text into 
numerics, and then, Lasso was applied to the data for classifying our mails into 
two different categories, spam, and ham.
As you know, Lasso is a kind of Regression model which is able to draw hyperplanes 
in order to classify our data, and also, because of having penalty term in its  
loss function, it accomplishes variable selection simultaneously, and this ML 
technique could be useful for NLP tasks.
'''

import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from nltk.corpus import stopwords;
import re
from nltk.stem import SnowballStemmer;
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;
from sklearn.metrics import confusion_matrix;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier;

text=pd.read_csv('...')


#text=pd.read_csv('',encoding='latin-1')
text.columns=['Category','Message']# Put some names on the columns
#Extract some variables (features) that might be useful for prediction

#Count the number of words in each message
text['word_count'] = text['Message'].agg(lambda x: len(x.split(" ")))


#Get the number of characters in each message
text['char_count'] = text['Message'].agg(lambda x:len(x))

#Remove the stopwprds
stop = stopwords.words('english')

#Change all the words to lowercase
text['Message']=text['Message'].agg(lambda x:x.lower())

#Number of stopwords used in the message can be an appropriate feature
text['stopwords'] = text['Message'].agg(lambda x: len([w for w in x.split() if w in stop]))

#Compare the features obtained so far for ham and spam messages
df2=text.groupby(text['Category'])
df2.agg('mean')
df2.agg('std')/np.sqrt(df2.agg('count').iloc[:,2:])


#Remove all charactes
text['Message'] = text['Message'].agg(lambda x:re.sub('[^\w\s]','',x))

#Stemming all words
stemmer = SnowballStemmer("english")
'''
'Stemming is the process of reducing a word to its word stem or to the roots 
of words known as a lemma. Stemming is important in natural language 
understanding (NLU) and natural language processing (NLP).'
'''

text['Message'] = text['Message'].agg(lambda x:(" ").join([stemmer.stem(w) for w in x.split()]))

spamtext=text[text['Category']=='spam']['Message']
hamtext=text[text['Category']=='ham']['Message']

#Get all words used in spam and ham mesagges (those words which are not stopwords)
spamwords =spamtext.agg(lambda x:[word for word in x.split() if word not in stop])
hamwords =hamtext.agg(lambda x:[word for word in x.split() if word not in stop])

#Join all these words and reconstruct the messages
spamwords=spamwords.agg(lambda x:' '.join(x))

#split the words again and count them across all the messages
spam_word_counts=spamwords.str.split(expand=True).stack().value_counts()

hamwords=hamwords.agg(lambda x:' '.join(x))
ham_word_counts=hamwords.str.split(expand=True).stack().value_counts()

#Select those words frequently used across all messages
spamvocab=spam_word_counts[spam_word_counts>=20]
hamvocab=ham_word_counts[ham_word_counts>=20]

s1=set(spamvocab.index)
Union=s1.union(set(hamvocab.index))
Union=pd.Series(list(Union))

allfeatures=np.zeros((text.shape[0],Union.shape[0]))
for i in np.arange(Union.shape[0]):
 allfeatures[:,i]=text['Message'].agg(lambda x:len(re.findall(Union[i],x)))


Complete_data=pd.concat([text,pd.DataFrame(allfeatures)],1)


X=Complete_data.iloc[:,2:]
y=Complete_data['Category']
enc=LabelEncoder()
enc.fit(y)
y = enc.transform(y)
repeat=50
acc_lasso_ham=np.empty(repeat)
acc_lasso_spam=np.empty(repeat)
acc_ridge_ham=np.empty(repeat)
acc_ridge_spam=np.empty(repeat)
acc_logistic_ham=np.empty(repeat)
acc_logistic_spam=np.empty(repeat)
acc_elnet_ham=np.empty(repeat)
acc_elnet_spam=np.empty(repeat)

for i in range(repeat):
    print(i)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logreg = LogisticRegression(solver='saga',penalty='none')
    lassologreg = LogisticRegression(C=15,penalty="l1",solver="liblinear")
    ridgelogreg = LogisticRegression(C=15,penalty="l2",solver="liblinear")
    #c same as t in slides
    elaslogreg=SGDClassifier(loss='log',penalty='elasticnet',alpha=0.0001,l1_ratio=1,tol=0.001)
    #l1_ratio equivalent to alpha in slides    
    #alpha same as lambda in slides
    lassologreg.fit(X_train,y_train)
    ridgelogreg.fit(X_train,y_train)
    logreg.fit(X_train,y_train)
    elaslogreg.fit(X_train,y_train)
    
    lasso=confusion_matrix(y_test,lassologreg.predict(X_test))
    ridge=confusion_matrix(y_test,ridgelogreg.predict(X_test))
    logistic=confusion_matrix(y_test,logreg.predict(X_test))
    elnet=confusion_matrix(y_test,elaslogreg.predict(X_test))
    
    
    acc_lasso_ham[i]=lasso[0,0]/sum(lasso[0,:])
    acc_lasso_spam[i]=lasso[1,1]/sum(lasso[1,:])
    acc_ridge_ham[i]=ridge[0,0]/sum(ridge[0,:])
    acc_ridge_spam[i]=ridge[1,1]/sum(ridge[1,:])
    acc_logistic_ham[i]=logistic[0,0]/sum(logistic[0,:])
    acc_logistic_spam[i]=logistic[1,1]/sum(logistic[1,:])
    acc_elnet_ham[i]=elnet[0,0]/sum(elnet[0,:])
    acc_elnet_spam[i]=elnet[1,1]/sum(elnet[1,:])

print('GLM Lasso Ham','\n',np.mean(acc_lasso_ham))
print('GLM Lasso Spam','\n',np.mean(acc_lasso_spam))
print('GLM Ridge Ham','\n',np.mean(acc_ridge_ham))
print('GLM Ridge Spam','\n',np.mean(acc_ridge_spam))
print('GLM Ham','\n',np.mean(acc_logistic_ham))
print('GLM Spam','\n',np.mean(acc_logistic_spam))
print('GLM Net Ham','\n',np.mean(acc_elnet_ham))
print('GLM Net Spam','\n',np.mean(acc_elnet_spam))
