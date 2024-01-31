# This is the Version 2 of the SMSSpamClassifier for deployment, the other version has all the details of the project

# The five steps to follow are DataCleaning, Exploratory Data Analysis, Data Pre-Processing, Model Building, Deployment
import pandas as pd
import numpy as np
import string
import pickle
import streamlit as st
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# download these if it raises an error i did because it raised an error
# nltk.download("punkt")
# nltk.download("stopwords")


def transforms_input(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)  # to tokenize the sentence
    x = []
    for i in sentence:  # to remove the special characters and appending only the alphanumeric charaters
        if i.isalnum() and i not in stopwords.words("english"):
            if i not in string.punctuation:
                x.append(pt.stem(i))  # here as i am appending it to x also i can pt.stem it so it can shorten the word

    return " ".join(x)  # returs the value in string format


pt = PorterStemmer()


# here unicodeDecodeError arises then set the encoding to ISO-8859-1
dataset = pd.read_csv("spam.csv", encoding="ISO-8859-1")
# here in info you can see alot of null columns
# dataset.info()
# print(dataset.isnull().sum())  # it gives the sum of null values in a column and according to that drop the columns

# ****** Data CLeaning ***********
# dropping the columns with most null values and inplace is set True for the making the changes permanent
dataset.drop(columns={"Unnamed: 2", "Unnamed: 3", "Unnamed: 4"}, inplace=True)

# Changing the column names to Target and Message
dataset.rename(columns={"v1": "Target", "v2": "Message"}, inplace=True)

# As you can see ham and spam representing in the columns change that to boolean 0 and 1 and for that import
# label encoder
encoder = LabelEncoder()
dataset["Target"] = encoder.fit_transform(dataset["Target"])

# to check duplicates
# print(dataset.duplicated().sum())
# now as duplicate values are so many drop them off, keep first leaves the first value and drops the rest of them
dataset = dataset.drop_duplicates(keep="first")

# ******** Exploratory Data Analysis ****
# this provides the value of 0's and 1's in the Target column
# print(dataset['Target'].value_counts())

# Creat a separate column for storing the no of characters in Message
# here the it counts the no of characters as len is applied and store in no_of_chars column
dataset["No_of_chars"] = dataset["Message"].apply(len)
# print(dataset.head(1)) # this prints the head of 1 elements in database

# Now in order to tokenize each word in the Message import nltk
# Creat another column to store the no of words in the Message
dataset["Words_count"] = dataset["Message"].apply(lambda x: len(nltk.word_tokenize(x)))
# Creat another column to store the no of sentences in the Message
dataset["Sentences_count"] = dataset["Message"].apply(lambda x: len(nltk.sent_tokenize(x)))


# **************** Data Preprocessing ***************
# 1 Lowercase
# 2 Tokenization
# 3 Remove Special characters
# 4 Remove Punctuations
# 5 Stemming
# did all this in the Transform function

# Now making a new Column to store the values of all these transformed messages
dataset["Transformed_message"] = dataset["Message"].apply(transforms_input)
# here i called the function and sent the message column as parameter


# ************8 Model Building ************
cv = CountVectorizer()
tfidf = TfidfVectorizer()
# X = cv.fit_transform(dataset["Transformed_message"]).toarray()  # checked with cv and then tfidf is better for this
X = tfidf.fit_transform(dataset["Transformed_message"]).toarray()

y = dataset["Target"].values
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# checking the scores of all
# gnb.fit(X_train, y_train)
# y_pred1 = gnb.predict(X_test)
# print(accuracy_score(y_test, y_pred1))
# print(confusion_matrix(y_test, y_pred1))
# print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
acc_score = accuracy_score(y_test, y_pred2)
conf_mat = confusion_matrix(y_test, y_pred2)
prec_score = precision_score(y_test, y_pred2)
# As mnb has a good precision score and accuracy score than others we are going with mnb

# bnb.fit(X_train, y_train)
# y_pred3 = bnb.predict(X_test)
# print(accuracy_score(y_test, y_pred3))
# print(confusion_matrix(y_test, y_pred3))
# print(precision_score(y_test, y_pred3))

# ******************** Now the actual project as the training is done **************************

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the Message")
if st.button("Predict"):
    # 1. Preprocess
    transformed_message = transforms_input(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_message])
    # 3. Predict
    result = mnb.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
    st.markdown(f"Accuracy scores are: {acc_score}")
    st.markdown(f"Confusion matrix is: {conf_mat}")
    st.markdown(f"Precision score is: {prec_score}")


