
class fake_News:
    #imports libraries
    from dataclasses import dataclass
    import numpy as np
    import pandas as pd
    import itertools
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    user_csv = input("What is the name of the csv file you want to analyze? ")
    df = pd.read_csv(user_csv)
    print("\nDimension of the dataset is " + str(df.shape))
    print("First 5 items in the dataset: " + str(df.head()))

    labels = df.label
    print("\nFirst 5 labels: " + str(labels.head()))

    #DataFlair - Split the dataset
    x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

    #Initialize a TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
    #Fit and transform train set, transform test set
    tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test=tfidf_vectorizer.transform(x_test)

    #Initialize a PassiveAggressiveClassifier
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train,y_train)

    #Predict on the test set and calculate accuracy
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print(f'\nAccuracy: {round(score*100,2)}%')

    #DataFlair - Build confusion matrix
    conf_mat = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
    print("\nConfusion Matrix: " + "\n" +str(conf_mat))



data = fake_News()
