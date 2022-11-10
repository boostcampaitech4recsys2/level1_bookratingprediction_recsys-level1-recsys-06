import streamlit as st
import pandas as pd
import numpy as np
import time

from data_load import context_data_load
from predict import single_prediction, top_k_prediction


def load_data()->pd.DataFrame:
    data = context_data_load()
    return data

st.title("Book Recommendation for Users")

with st.spinner("책 로딩중..."):
    if 'data' not in st.session_state:
        st.session_state.data = load_data()
    st.dataframe(st.session_state.data['books'][['isbn','book_title','book_author','category']])

# Online Recommendation
st.header("Online Recommendation")
st.subheader("예측하고 싶은 유저와 책을 입력하세요.")
with st.form(key="online 입력 form"):
    user_id = st.number_input("유저의 ID",0,68068,step=1)
    isbn = st.number_input("책의 ISBN",0,149569,step=1)

    submitted = st.form_submit_button("예측 시작!")
    if submitted:
        st.table(single_prediction(st.session_state.data,user_id,isbn))

# Batch Recommendation
st.header("Batch Recommendation")
st.subheader("예측하고 싶은 유저와 Top K값을 입력하세요.")
with st.form(key="batch 입력 form"):
    user_id = st.number_input("유저의 ID",0,68068,step=1)
    top_k = st.number_input("Top K",1,149570,step=1)

    submitted = st.form_submit_button("예측 시작!")
    if submitted:
        st.table(top_k_prediction(st.session_state.data,user_id,top_k))

