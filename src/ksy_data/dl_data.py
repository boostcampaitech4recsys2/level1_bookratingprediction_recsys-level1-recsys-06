import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def make_others(train, test, _column, n):

    tem = pd.DataFrame(train[_column].value_counts()).reset_index()
    tem.columns = ['names','count']
    others_list = tem[tem['count'] <= n]['names'].values  # n은 초기에 설정함. 바꿀 수 있음.
    train.loc[train[train[_column].isin(others_list)].index, _column]= 'otehrs'
    test.loc[test[test[_column].isin(others_list)].index, _column]= 'otehrs'
    return train, test

def dl_data_load(args):

    ######################## DATA LOAD
    train = pd.read_csv(args.DATA_PATH + 'ksy_train_rating_fianl1.csv')
    test = pd.read_csv(args.DATA_PATH + 'ksy_test_rating_fianl1.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    train, test = make_others(train, test, 'user_id', args.USER_N)
    train, test = make_others(train, test, 'isbn', args.ISBN_N)

    _data = pd.concat([train, test])

    idx2user = {idx:id for idx, id in enumerate(_data['user_id'].unique())}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(_data['isbn'].unique())}

    user2idx = {id:idx for idx, id in enumerate(_data['user_id'].unique())}
    isbn2idx = {isbn:idx for idx, isbn in enumerate(_data['isbn'].unique())}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    train = train[['user_id', 'isbn', 'rating']]
    test = test[['user_id', 'isbn', 'rating']]

    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'sub':sub,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            }


    return data

def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
