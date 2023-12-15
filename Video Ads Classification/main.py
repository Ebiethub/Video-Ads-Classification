import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import numpy as np
import pandas as pd
import streamlit as st
import pickle

tokenizer = pickle.load(open('video_classification-tokenizer.pkl', 'rb'))
model = load_model('video_classification-model.h5')

st.title('Video Ads Classification')
st.subheader('Description')
st.write('The model uses the title and description of a video to predict if it belongs')
st.write('to the following class listed below:')
st.write('The classes are; science and technology, food, manufacturing, travel, history, art and music ', )
st.subheader('Sample texts to use')
st.write('<h6>Title: Painting UNBELIEVABLE Halloween wall | Ft. Smoe</h6>', unsafe_allow_html=True)
st.write('<h6>Description: We created this epic Halloween painting in the Garden of my friend Smoe. It is one of the </h6>', unsafe_allow_html=True)
st.write('<p>You can also try it out with your own random text and see the outcome</p>', unsafe_allow_html=True)
title = st.text_input('Video Title')
desc = st.text_input('Video Description')

submit = st.button('Predict')
#On predict button click
if submit:
    def predict(title, desc):
        # Max number of words in each complaint.
        MAX_SEQUENCE_LENGTH = 50
        data_for_lstms = []
        data_for_lstms.append(' '.join([title, desc]))
        # Convert the data to padded sequences
        X = tokenizer.texts_to_sequences(data_for_lstms)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        predict_x = model.predict(X)
        classes_x = np.argmax(predict_x, axis=1)
        return classes_x


    def label_reverse(result):
        if result == 0:
            return 'Art and Music'
        elif result == 1:
            return 'Food'
        if result == 2:
            return 'History'
        if result == 3:
            return 'Manufacturing'
        if result == 4:
            return 'Science and Technology'
        else:
            return 'Travel'


    st.title(label_reverse(predict(title, desc)))
