import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import pickle


def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    
    
    radio_menu=['데이터프레임', '통계치','상관분석표']
    selected_radio = st.radio('선택하세요',radio_menu)
    
    if selected_radio == '데이터프레임' :
        st.dataframe(car_df)

        

    elif selected_radio == '통계치' :
        st.dataframe(car_df.describe())

    columns = car_df.columns
    columns = list(columns)

    selected_columns = st.multiselect('컬럼을 선택하시오', columns)
    
    if len(selected_columns) != 0 :
        st.dataframe( car_df[selected_columns])
    else :
        st.write('선택한 컬럼이 없습니다.')

    #상관 계수 화면을 보여주도록 만듬.
    #멀티셀렉트에 컬럼명을 보여주고,
    #해당 컬럼들에 대한 상관계수를 보여주세요.
    #단 컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야함.

    
    corr_columns = car_df.columns [car_df.dtypes != object ]
    selected_corr = st.multiselect('상관계수 컬럼선택', corr_columns)

    if len(selected_corr) > 0 :
        st.dataframe( car_df[ selected_corr ].corr() )          ## 위에서 선택한 컬럼들을 이용해서, pairplot 차트 그리기.
        
        
        fig=sns.pairplot(data = car_df[selected_corr] )
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')
    
    # 컬럼을 하나만 선택하면, 해당 컬럼의 man와 min에 해당하는 사람의 데이터를 화면에 보여주는 기능.(숫자로 된 데이터만.)
    number_columns = car_df.columns [car_df.dtypes != object ]
    min_max = st.selectbox('컬럼을 선택하세요.', number_columns)
    
    if len(min_max) != 0 :
        st.write('최소')
        st.dataframe( car_df.loc[car_df[min_max] == car_df[min_max].min(),  ] )
        st.write('최대')
        st.dataframe( car_df.loc[car_df[min_max] == car_df[min_max].max(),  ] )
    else :
        st.write('선택한 컬럼이 없습니다.')


    #고객이름을 검색할 수 있는 기능 개발.
    
    word = st.text_input('고객이름검색')
    
    if len(word) != 0 :
        
        st.dataframe( car_df.loc[ car_df['Customer Name'].str.contains(word, case=False), ] )
    else:
        st.write('선택한 컬럼이 없습니다.')


    
    
    