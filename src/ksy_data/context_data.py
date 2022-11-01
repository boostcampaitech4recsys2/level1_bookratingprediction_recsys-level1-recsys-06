import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


# 전처리된 데이터 로드
def context_data_load(args):

    ######################## DATA LOAD
    train = pd.read_csv(args.DATA_PATH + 'ksy_train_rating_10n.csv')
    test = pd.read_csv(args.DATA_PATH + 'ksy_test_rating_10n.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    data = pd.concat([train, test])

    ### 라벨 인코딩 과정
    user2idx = {id:idx for idx, id in enumerate(data['user_id'].unique())}
    isbn2idx = {isbn:idx for idx, isbn in enumerate(data['isbn'].unique())}

    author2idx = {author:idx for idx, author in enumerate(data['book_author'].unique())}
    publisher2idx = {publisher:idx for idx, publisher in enumerate(data['publisher'].unique())}
    language2idx = {language:idx for idx, language in enumerate(data['language'].unique())}
    category2idx = {category:idx for idx, category in enumerate(data['category_high'].unique())}
    year2idx = {year:idx for idx, year in enumerate(data['years'].unique())}
    location_state2idx = {location_state:idx for idx, location_state in enumerate(data['fix_location_state'].unique())}
    age2idx = {age:idx for idx, age in enumerate(data['fix_age'].unique())}

    idx2user = {id:idx for idx, id in user2idx.items()}
    idx2isbn = {isbn:idx for idx, isbn in isbn2idx.items()} 

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    train['book_author'] = train['book_author'].map(author2idx)
    test['book_author'] = test['book_author'].map(author2idx)

    train['publisher'] = train['publisher'].map(publisher2idx)
    test['publisher'] = test['publisher'].map(publisher2idx)

    train['language'] = train['language'].map(language2idx)
    test['language'] = test['language'].map(language2idx)

    train['category_high'] = train['category_high'].map(category2idx)
    test['category_high'] = test['category_high'].map(category2idx)

    train['years'] = train['years'].map(year2idx)
    test['years'] = test['years'].map(year2idx)

    train['fix_location_state'] = train['fix_location_state'].map(location_state2idx)
    test['fix_location_state'] = test['fix_location_state'].map(location_state2idx)

    train['fix_age'] = train['fix_age'].map(age2idx)
    test['fix_age'] = test['fix_age'].map(age2idx)

    field_dims = np.array([len(user2idx), len(isbn2idx), len(author2idx),
                            len(publisher2idx), len(language2idx), len(category2idx),
                            len(year2idx), len(location_state2idx), len(age2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn
            }


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
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

# 데이터 로더
def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
