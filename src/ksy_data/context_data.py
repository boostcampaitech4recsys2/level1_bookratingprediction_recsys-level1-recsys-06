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

# 전처리된 데이터 로드
def context_data_load(args):

    ######################## DATA LOAD
    train = pd.read_csv(args.DATA_PATH + 'ksy_train_rating_fianl1.csv')
    test = pd.read_csv(args.DATA_PATH + 'ksy_test_rating_fianl1.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    

    # DCN과 FFM 적용시 Others를 다르게 적용해야해서
    train['user_id_D'] = train['user_id'].copy()
    test['user_id_D'] = test['user_id'].copy()
    train['user_id_F'] = train['user_id'].copy()
    test['user_id_F'] = test['user_id'].copy()

    train['isbn_D'] = train['isbn'].copy()
    test['isbn_D'] = test['isbn'].copy()
    train['isbn_F'] = train['isbn'].copy()
    test['isbn_F'] = test['isbn'].copy()

    train.drop(['user_id', 'isbn'], axis = 1, inplace = True)
    test.drop(['user_id', 'isbn'], axis = 1, inplace = True)

    # others 삽입
    train, test = make_others(train, test, 'user_id_D', args.USER_N_D)
    train, test = make_others(train, test, 'user_id_F', args.USER_N_F)
    
    train, test = make_others(train, test, 'isbn_D', args.ISBN_N_D)
    train, test = make_others(train, test, 'isbn_F', args.ISBN_N_F)

    train, test = make_others(train, test, 'book_author', args.AUTHOR_N)
    train, test = make_others(train, test, 'publisher', args.PUBLISH_N)
    train, test = make_others(train, test, 'category_high', args.CATEGORY_N)
    train, test = make_others(train, test, 'location_state', args.STATE_N)
    train, test = make_others(train, test, 'location_country', args.COUNTRY_N)
    train, test = make_others(train, test, 'location_city', args.CITY_N)

    _data = pd.concat([train, test])

    ### 라벨 인코딩 과정
    user2Didx = {id:idx for idx, id in enumerate(_data['user_id_D'].unique())}
    isbn2Didx = {isbn:idx for idx, isbn in enumerate(_data['isbn_D'].unique())}

    user2Fidx = {id:idx for idx, id in enumerate(_data['user_id_F'].unique())}
    isbn2Fidx = {isbn:idx for idx, isbn in enumerate(_data['isbn_F'].unique())}

    # idx2user = {idx:id for idx, id in enumerate(_data['user_id'].unique())}
    # idx2isbn = {idx:isbn for idx, isbn in enumerate(_data['isbn'].unique())}

    author2idx = {author:idx for idx, author in enumerate(_data['book_author'].unique())}
    publisher2idx = {publisher:idx for idx, publisher in enumerate(_data['publisher'].unique())}
    language2idx = {language:idx for idx, language in enumerate(_data['language'].unique())}
    category2idx = {category:idx for idx, category in enumerate(_data['category_high'].unique())}
    year2idx = {year:idx for idx, year in enumerate(_data['years'].unique())}
    location_state2idx = {location_state:idx for idx, location_state in enumerate(_data['location_state'].unique())}
    location_city2idx = {location_city:idx for idx, location_city in enumerate(_data['location_city'].unique())}
    location_country2idx = {location_country:idx for idx, location_country in enumerate(_data['location_country'].unique())}
    age2idx = {age:idx for idx, age in enumerate(_data['fix_age'].unique())}

    train['user_id_D'] = train['user_id_D'].map(user2Didx)
    train['user_id_F'] = train['user_id_F'].map(user2Fidx)
    #sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id_D'] = test['user_id_D'].map(user2Didx)
    test['user_id_F'] = test['user_id_F'].map(user2Fidx)

    train['isbn_D'] = train['isbn_D'].map(isbn2Didx)
    train['isbn_F'] = train['isbn_F'].map(isbn2Fidx)
    #sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn_D'] = test['isbn_D'].map(isbn2Didx)
    test['isbn_F'] = test['isbn_F'].map(isbn2Fidx)

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

    train['location_state'] = train['location_state'].map(location_state2idx)
    test['location_state'] = test['location_state'].map(location_state2idx)

    train['location_city'] = train['location_city'].map(location_city2idx)
    test['location_city'] = test['location_city'].map(location_city2idx)

    train['location_country'] = train['location_country'].map(location_country2idx)
    test['location_country'] = test['location_country'].map(location_country2idx)

    train['fix_age'] = train['fix_age'].map(age2idx)
    test['fix_age'] = test['fix_age'].map(age2idx)

    train = train[['user_id_D','isbn_D','user_id_F','isbn_F','book_author','publisher','language','category_high','years','location_city','location_state','location_country','fix_age','rating']]
    test = test[['user_id_D','isbn_D','user_id_F','isbn_F','book_author','publisher','language','category_high','years','location_city','location_state','location_country','fix_age','rating']]
    field_dims = np.array([len(user2Didx), len(isbn2Didx), len(user2Fidx), len(isbn2Fidx), len(author2idx),
                            len(publisher2idx), len(language2idx), len(category2idx),
                            len(year2idx), len(location_city2idx), len(location_state2idx), 
                            len(location_country2idx), len(age2idx)], dtype=np.uint32)


    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'sub':sub,
            # 'idx2user':idx2user,
            # 'idx2isbn':idx2isbn,
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

# valid 없이 가보자.
# def context_data_loader(args, data):
#     train_dataset = TensorDataset(torch.LongTensor(data['train'].drop(['rating'], axis=1).values), torch.LongTensor(data['rating'].values))
#     test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

#     train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

#     data['train_dataloader'], data['test_dataloader'] = train_dataloader, test_dataloader

#     return data
