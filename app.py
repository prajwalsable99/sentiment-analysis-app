import streamlit as st
import os
import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import joblib

import warnings
 
warnings.filterwarnings("ignore")

@st.cache
def make_prediction(review):
 
 
 
   
    model = joblib.load("my_model.joblib")
 
    # make prection
    result = model.predict([review])
 
    
 
    flag=result[0]
    if(flag==1):
        return "positive review"
    else:
        return "negative review"

# print(make_prediction("Would not go back"))

st.title("restaurant review classifier App")
st.write(
    "A simple machine laerning app to predict the sentiment of a restaurant's review"
)

form = st.form(key="my_form")
review = form.text_input(label="Enter the text of your restaurant review")
submit = form.form_submit_button(label="Make Prediction")

if submit:
    # make prediction from the input text
    result= make_prediction(review)
 
    # Display results of the NLP task
    st.header("Results")
 
  
    st.write("review is : ", result)
    
        