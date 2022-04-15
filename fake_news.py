#imports libraries 
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split #responsible for splitting the data set into training data and test data
from sklearn.feature_extraction.text import TfidfVectorizer #converts raw documents into a matrix of TF-IDF features
from sklearn.linear_model import PassiveAggressiveClassifier #online learning algorithm which makes updates based on losses (Passive if their is a correct classifcation, Aggressive (updates and adjusts) when their is a miscalculaltion)
from sklearn.metrics import accuracy_score, confusion_matrix #evaluates how well our model works

#using pandas reads in the news data
df = pd.read_csv('news.csv')
#using the .head() to see how our data looks and the dimensions of our data. The output will be the first 5 objects of our data and a tuple representing the dimensions of the data
df.shape
df.head()

#gets the labels of each news article which represents if it is Fake News or Not
labels = df.label
labels.head()

#splits the data set into training and test data (takes in the text columns and the labels, and determines the proportion of the data set to be included to be 0.2, random state controls the shuffling that is applied to the dataset before the split )
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidVectorizer(takes in english as a stop word which will remove all the "english" text in the document. stop words are used to filter out words not are not important and may intefere with our model, max_df removes terms that appear to frequently. In this case we remove any items that appear more than 70% of the documents)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set(trains our model), transform test set (uses the same mean and variance as it is calculated from our training data to transform our test data), we do not fit_transform our test data because we want it to be a suprise for the model to mantain an unbias model.
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier(max_iter defines the maximum number of passes over the training data, in this case we pass the training data 50 times)
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train) #fits the linear model with the Passive Agressive Algorithmn

#Predict on the test set and calculate accuracy. The accuracy is based on how well our predicted model (on test data) works that is how well our labels match to the test data. The ouput of score will be a value from 0 to 1 representing the percentage our predict data mathced with the test data
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix which evaluates the accuracy of a classification(the true positives, true negatives, false positives, and false negatives)
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
