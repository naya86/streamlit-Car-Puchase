import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import pickle
import joblib


def run_ml_app():
    st.subheader('Machine Learning')

    #1. 유저에게 입력 받는다.
    # 
    
    gender = st.radio('성별을 선택하세요.', ['남자', '여자'] )

    if gender == '남자' :
        gender = 1
    else :
        gender = 0

    age = st.number_input('나이를 입력하세요.', min_value=1, max_value=120)

    salary = st.number_input('연봉입력', min_value=0)

    debt = st.number_input('빚 입력', min_value=0)

    worth = st.number_input('자산', min_value=0)
    
    
    #2. 예측하기.
    
    #2-1모델 불러오기

    model = tensorflow.keras.models.load_model('data/check.h5')

    # 2-2 넘파이 어레이로 만들기

    new_data = np.array( [ gender, age, salary, debt, worth ] )

    # 2-3 피쳐스케일링

    new_data = new_data.reshape(1,-1)  # 피쳐스케일링은 2차원으로 들어가야함.

    sc_x = joblib.load('data/sc_X.pkl') #스케일러 불러오기
    
    new_data = sc_x.transform(new_data) #피쳐스케일링

    # 2-4 예측
    
    y_pred = model.predict(new_data)  #예측.

    #예측결과는 스케일링 된 결과이므로 다시 돌려야한다.
    # st.write(y_pred[0][0])  

    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform(y_pred)
    #st.write(y_pred_original)   #예측결과
   
   
    # 3. 결과를 화면에 보여줌
    btn = st.button('결과보기')
    if btn :
        st.write('예측결과입니다. {:,.1f}달러의 차를 살 수 있습니다.'.format(y_pred_original[0,0]))
    
    