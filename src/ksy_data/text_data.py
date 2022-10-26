import os
import re
import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer


def text_preprocessing(summary):
    summary = re.sub("[.,\'\"''""!?]", "", summary)
    summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
    summary = re.sub("\s+", " ", summary)
    summary = summary.lower()
    return summary


def summary_merge(df, user_id, max_summary):
    return " ".join(df[df['user_id'] == user_id].sort_values(by='summary_length', ascending=False)['summary'].values[:max_summary])


def text_to_vector(text, tokenizer, model, device):
    for sent in tokenize.sent_tokenize(text):
        text_ = "[CLS] " + sent + " [SEP]"
        tokenized = tokenizer.tokenize(text_)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segments_idx = [1] * len(tokenized)
        token_tensor = torch.tensor([indexed])
        sgments_tensor = torch.tensor([segments_idx])
        with torch.no_grad():
            outputs = model(token_tensor.to(device), sgments_tensor.to(device))
            encode_layers = outputs[0]
            sentence_embedding = torch.mean(encode_layers[0], dim=0)
    return sentence_embedding.cpu().detach().numpy()


def process_text_data(df, books, user2idx, isbn2idx, device, train=False, user_summary_merge_vector=False, item_summary_vector=False):
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx)

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx)
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'summary']], on='isbn', how='left')
    df_['summary'].fillna('None', inplace=True)
    df_['summary'] = df_['summary'].apply(lambda x:text_preprocessing(x))
    df_['summary'].replace({'':'None', ' ':'None'}, inplace=True)
    df_['summary_length'] = df_['summary'].apply(lambda x:len(x))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    if user_summary_merge_vector and item_summary_vector:
        print('Create User Summary Merge Vector')
        user_summary_merge_vector_list = []
        for user in tqdm(df_['user_id'].unique()):
            vector = text_to_vector(summary_merge(df_, user, 5), tokenizer, model, device)
            user_summary_merge_vector_list.append(vector)
        user_review_text_df = pd.DataFrame(df_['user_id'].unique(), columns=['user_id'])
        user_review_text_df['user_summary_merge_vector'] = user_summary_merge_vector_list
        vector = np.concatenate([
                                user_review_text_df['user_id'].values.reshape(1, -1),
                                user_review_text_df['user_summary_merge_vector'].values.reshape(1, -1)
                                ])
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_user_summary_merge_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_user_summary_merge_vector.npy', vector)

        print('Create Item Summary Vector')
        item_summary_vector_list = []
        books_text_df = df_[['isbn', 'summary']].copy()
        books_text_df= books_text_df.drop_duplicates().reset_index(drop=True)
        books_text_df['summary'].fillna('None', inplace=True)
        for summary in tqdm(books_text_df['summary']):
            vector = text_to_vector(summary, tokenizer, model, device)
            item_summary_vector_list.append(vector)
        books_text_df['item_summary_vector'] = item_summary_vector_list
        vector = np.concatenate([
                                books_text_df['isbn'].values.reshape(1, -1),
                                books_text_df['item_summary_vector'].values.reshape(1, -1)
                                ])
        if not os.path.exists('./data/text_vector'):
            os.makedirs('./data/text_vector')
        if train == True:
            np.save('./data/text_vector/train_item_summary_vector.npy', vector)
        else:
            np.save('./data/text_vector/test_item_summary_vector.npy', vector)
    else:
        print('Check Vectorizer')
        print('Vector Load')
        if train == True:
            user = np.load('data/text_vector/train_user_summary_merge_vector.npy', allow_pickle=True)
        else:
            user = np.load('data/text_vector/test_user_summary_merge_vector.npy', allow_pickle=True)
        user_review_text_df = pd.DataFrame([user[0], user[1]]).T
        user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
        user_review_text_df['user_id'] = user_review_text_df['user_id'].astype('int')

        if train == True:
            item = np.load('data/text_vector/train_item_summary_vector.npy', allow_pickle=True)
        else:
            item = np.load('data/text_vector/test_item_summary_vector.npy', allow_pickle=True)
        books_text_df = pd.DataFrame([item[0], item[1]]).T
        books_text_df.columns = ['isbn', 'item_summary_vector']
        books_text_df['isbn'] = books_text_df['isbn'].astype('int')


    df_ = pd.merge(df_, user_review_text_df, on='user_id', how='left')
    df_ = pd.merge(df_, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left')

    return df_


class Text_Dataset(Dataset):
    def __init__(self, user_isbn_vector, user_summary_merge_vector, item_summary_vector, label):
        self.user_isbn_vector = user_isbn_vector
        self.user_summary_merge_vector = user_summary_merge_vector
        self.item_summary_vector = item_summary_vector
        self.label = label

    def __len__(self):
        return self.user_isbn_vector.shape[0]

    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'user_summary_merge_vector' : torch.tensor(self.user_summary_merge_vector[i].reshape(-1, 1), dtype=torch.float32),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def text_data_load(args):

    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)

    text_train = process_text_data(train, books, user2idx, isbn2idx, args.DEVICE, train=True, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE)
    text_test = process_text_data(test, books, user2idx, isbn2idx, args.DEVICE, train=False, user_summary_merge_vector=args.DEEPCONN_VECTOR_CREATE, item_summary_vector=args.DEEPCONN_VECTOR_CREATE)

    data = {
            'train':train,
            'test':test,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'text_train':text_train,
            'text_test':text_test,
            }

    return data


def text_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['text_train'][['user_id', 'isbn', 'user_summary_merge_vector', 'item_summary_vector']],
                                                        data['text_train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


def text_data_loader(args, data):
    train_dataset = Text_Dataset(
                                data['X_train'][['user_id', 'isbn']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['item_summary_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Text_Dataset(
                                data['X_valid'][['user_id', 'isbn']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['item_summary_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Text_Dataset(
                                data['text_test'][['user_id', 'isbn']].values,
                                data['text_test']['user_summary_merge_vector'].values,
                                data['text_test']['item_summary_vector'].values,
                                data['text_test']['rating'].values
                                )


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
