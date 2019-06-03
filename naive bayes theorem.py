import os 
import nltk
import nltk.corpus
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
 
#printing all the corpus
print (os.listdir(nltk.data.find("corpora")))

#building a machine learning classifier on one of the corpus
#lets take the movie review

from nltk.corpus import movie_reviews

print (movie_reviews.categories())
pos_rev = movie_reviews.fileids('pos')
print (len(pos_rev))
neg_rev = movie_reviews.fileids('neg')
print (len(neg_rev))

#using one of the reviews from the positive reviews section
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
print (rev)

rev_list =[]
for rev in neg_rev:
    rev_text_neg = rev = nltk.corpus.movie_reviews.words(rev)
    review_one_string = " ".join(rev_text_neg)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .', '.')
    review_one_string = review_one_string.replace(' \'',"'")
    review_one_string = review_one_string.replace('\'', "'")
    rev_list.append(review_one_string)
    
print (len(rev_list))

for rev_pos in pos_rev:
    rev_text_pos = rev = nltk.corpus.movie_reviews.words(rev_pos)
    review_one_string = " ".join(rev_text_pos)
    review_one_string = review_one_string.replace(' ,', ',')
    review_one_string = review_one_string.replace(' .', '.')
    review_one_string = review_one_string.replace(' \'',"'")
    review_one_string = review_one_string.replace('\'', "'")
    rev_list.append(review_one_string)
    
print (len(rev_list))

#setting up the targets
neg_targets = np.zeros((1000,),dtype=np.int)
pos_targets = np.ones((1000,),dtype = np.int)
target_list = []
for i in neg_targets:
    target_list.append(i)
for j in pos_targets:
    target_list.append(j)
    
print (len(target_list))

#intializing pandas series
y = pd.Series(target_list)
print (type(y))

print (y.head())
count_vect = CountVectorizer(lowercase=True,stop_words='english',min_df = 2)
X_count_vect = count_vect.fit_transform(rev_list)
print (X_count_vect.shape)

X_names = count_vect.get_feature_names()
print(X_names)
X_count_vect = pd.DataFrame(X_count_vect.toarray(),columns = X_names)
print(X_count_vect.shape)

print (X_count_vect.head())

from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.metrics import confussion_matrix

X_train_cv,X_test_cv,y_train_cv,y_test_cv = train_test_split(X_count_vect,y,test_size=0.25,random_state = 5)

print(X_train_cv.shape)
print ("-------------------------------------------------------------------")
print (X_test_cv.shape)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

clf_cv = MultinomialNB()
clf_cv.fit(X_train_cv,y_train_cv)

y_pred_cv = clf_cv.predict(X_test_cv)
print (y_pred_cv)

import matplotlib.pyplot as plt

b = pd.DataFrame(y_test_cv,y_pred_cv)
print (b.shape)

plot = plt.Circle(b)
print (plt.show(plot))

    