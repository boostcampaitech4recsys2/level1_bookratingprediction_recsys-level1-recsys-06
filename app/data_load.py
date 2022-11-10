import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pickle

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(data, user_id, book_id=0, batch=False):
    users = data['users']
    users = users.drop(['location'], axis=1)
    books = data['books']
    
    if batch:
        ratings2 = pd.DataFrame({
            'user_id':[user_id]*len(data['isbn2idx']),
            'isbn':data['isbn2idx'].values()
        })
    else:
        ratings2 = pd.DataFrame({
            'user_id':[user_id],
            'isbn':[book_id]
        })


    # 인덱싱 처리된 데이터 조인
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')

    # 인덱싱 처리
    loc_city2idx = data['idx']['loc_city2idx']
    loc_state2idx =  data['idx']['loc_state2idx']
    loc_country2idx =  data['idx']['loc_country2idx']

    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    test_df['age'] = test_df['age'].fillna(30)
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = data['idx']['category2idx']
    publisher2idx = data['idx']['publisher2idx']
    language2idx = data['idx']['language2idx']
    author2idx = data['idx']['author2idx']

    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    data['test'] = test_df
    return data

# 전처리된 데이터 로드
def context_data_load():

    ######################## DATA LOAD
    with open('data_process.pkl','rb') as f:
        data = pickle.load(f)
    return data

# 데이터 split
def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
                                                        
    print(args.SEED)
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

# 데이터 로더
def context_data_loader(data):
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    data['test_dataloader'] = test_dataloader

    return data

if __name__ == '__main__':
    data = context_data_load()
    data = process_context_data(data,0,0)
    print(data['test'])
    context_data_loader(data)
    print(next(iter(data['test_dataloader'])))