import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

path= 'data/'

users = pd.read_csv(path+'cat_users.csv')
books = pd.read_csv(path+'cat_books.csv')

st.title('Predict User gives a score to a Book')
# selected_item = st.radio("Radio Part", ("A", "B", "C"))
text_input1 = int(st.text_input("User ID 입력 하세요. (ex : 8, 278854)"))
id = users[users['user_id'] == text_input1].reset_index(drop=True)
text_input2 = st.text_input("Books ISBN 입력 하세요. (ex : 0002005018)")
isbn = books[books['isbn'] == text_input2].reset_index(drop=True)

dat = pd.concat([id,isbn], axis=1)
dat = dat[['user_id','isbn','book_title','book_author','publisher','language','category_high','years','location_city','location_state','location_country','fix_age']]
dat = dat.astype({'user_id':'str', 'years':'str', 'location_city':'str', 'location_state':'str', 'location_country':'str', 'fix_age':'str'})

loaded_model = joblib.load('MODEL/Cat_model.pkl')
pred = loaded_model.predict(dat)
pred = np.round(pred[0], 2)
st.write(f'유저 {text_input1}님에 책 {text_input2} 에 대한 예측 평점은 {pred}입니다.')