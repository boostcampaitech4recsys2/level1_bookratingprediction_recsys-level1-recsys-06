import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold



def make_others(data_f, _column, n):

    tem = pd.DataFrame(data_f[_column].value_counts()).reset_index()
    tem.columns = ['names','count']
    others_list = tem[tem['count'] <= n]['names'].values  # n은 초기에 설정함. 바꿀 수 있음.
    data_f.loc[data_f[data_f[_column].isin(others_list)].index, _column]= 'others'
    return data_f


def make_others2(train, test, _column, n):

    tem = pd.DataFrame(train[_column].value_counts()).reset_index()
    tem.columns = ['names','count']
    others_list = tem[tem['count'] <= n]['names'].values  # n은 초기에 설정함. 바꿀 수 있음.
    train.loc[train[train[_column].isin(others_list)].index, _column]= 'others'
    test.loc[test[test[_column].isin(others_list)].index, _column]= 'others'
    return train, test

# 전처리된 데이터 로드
def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'ksy_users_fianl2.csv')
    books = pd.read_csv(args.DATA_PATH + 'ksy_books_fianl2.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    users = make_others(users, 'location_city', args.OTHERS_N)
    users = make_others(users, 'location_state', args.OTHERS_N)
    users = make_others(users, 'location_country', args.OTHERS_N)

    books = make_others(books, 'book_author', args.OTHERS_N)
    books = make_others(books, 'publisher', args.OTHERS_N)
    books = make_others(books, 'category_high', args.OTHERS_N)

    train = pd.merge(train,books, how='right',on='isbn')
    train.dropna(subset=['rating'], inplace = True)
    train = pd.merge(train, users, how='right',on='user_id')
    train.dropna(subset=['rating'], inplace = True)

    test['index'] = test.index
    test = pd.merge(test,books, how='right',on='isbn')
    test.dropna(subset=['rating'], inplace = True)
    test = pd.merge(test, users, how='right',on='user_id')
    test.dropna(subset=['rating'], inplace = True)
    test = test.sort_values('index')
    test.drop(['index'], axis=1, inplace=True)

    # print(train.shape)
    # print(test.head(3))

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
    train, test = make_others2(train, test, 'user_id_D', args.USER_N_D)
    train, test = make_others2(train, test, 'user_id_F', args.USER_N_F)
    
    train, test = make_others2(train, test, 'isbn_D', args.ISBN_N_D)
    train, test = make_others2(train, test, 'isbn_F', args.ISBN_N_F)

    _data = pd.concat([train, test])

    ### 라벨 인코딩 과정
    user2Didx = {id:idx for idx, id in enumerate(_data['user_id_D'].unique())}
    isbn2Didx = {isbn:idx for idx, isbn in enumerate(_data['isbn_D'].unique())}

    user2Fidx = {id:idx for idx, id in enumerate(_data['user_id_F'].unique())}
    isbn2Fidx = {isbn:idx for idx, isbn in enumerate(_data['isbn_F'].unique())}

    idx2user2D = {idx:id for idx, id in enumerate(_data['user_id_D'].unique())}
    idx2isbn2D = {idx:isbn for idx, isbn in enumerate(_data['isbn_D'].unique())}

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
            'idx2user':idx2user2D,
            'idx2isbn':idx2isbn2D,
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.SEED)
    folds = []
    for train_idx, valid_idx in skf.split(data['train'].drop(['rating'], axis = 1), data['train']['rating']):
        folds.append((train_idx,valid_idx))

    for fold in range(0,5):
        train_idx, valid_idx = folds[fold]
        data[f'X_train{fold}'] = data['train'].drop(['rating'],axis = 1).iloc[train_idx]
        data[f'X_valid{fold}'] = data['train'].drop(['rating'],axis = 1).iloc[valid_idx]
        data[f'y_train{fold}'] = data['train']['rating'].iloc[train_idx]
        data[f'y_valid{fold}'] = data['train']['rating'].iloc[valid_idx]

    return data

# 데이터 로더
def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    for fold in range(0,5):
        train_dataset = TensorDataset(torch.LongTensor(data[f'X_train{fold}'].values), torch.LongTensor(data[f'y_train{fold}'].values))
        valid_dataset = TensorDataset(torch.LongTensor(data[f'X_valid{fold}'].values), torch.LongTensor(data[f'y_valid{fold}'].values))
        train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
        data[f'train_dataloader{fold}'], data[f'valid_dataloader{fold}'] = train_dataloader, valid_dataloader
    

    return data

# #valid 없이 가보자.
# def context_data_loader(args, data):
#     train_dataset = TensorDataset(torch.LongTensor(data['train'].drop(['rating'], axis=1).values), torch.LongTensor(data['train']['rating'].values))
#     test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

#     train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

#     data['train_dataloader'], data['test_dataloader'] = train_dataloader, test_dataloader

#     return data
