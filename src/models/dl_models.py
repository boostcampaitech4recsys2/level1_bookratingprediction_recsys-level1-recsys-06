import tqdm

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _NeuralCollaborativeFiltering, _WideAndDeepModel, _DeepCrossNetworkModel, _FFDCNModel
from ._models import rmse, RMSELoss

class NeuralCollaborativeFiltering:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx=np.array((1, ), dtype=np.long)

        self.embed_dim = args.NCF_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.NCF_MLP_DIMS
        self.dropout = args.NCF_DROPOUT

        self.model = _NeuralCollaborativeFiltering(self.field_dims, user_field_idx=self.user_field_idx, item_field_idx=self.item_field_idx,
                                                    embed_dim=self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class WideAndDeepModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.WDN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.WDN_MLP_DIMS
        self.dropout = args.WDN_DROPOUT

        self.model = _WideAndDeepModel(self.field_dims, self.embed_dim, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class DeepCrossNetworkModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = _DeepCrossNetworkModel(self.field_dims, self.embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class FFDCNModel:

    def __init__(self, args, dataffm):
        super().__init__()

        self.criterion = RMSELoss()

        self.ff_train_dataloader = dataffm['train_dataloader']
        self.ff_valid_dataloader = dataffm['valid_dataloader']
        self.ff_field_dims = dataffm['field_dims']
        # self.dcn_train_dataloader = datadcn['train_dataloader']
        # self.dcn_valid_dataloader = datadcn['valid_dataloader']
        # self.dcn_field_dims = datadcn['field_dims']

        self.ff_embed_dim = args.FM_EMBED_DIM
        self.dcn_embed_dim = args.DCN_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        
        self.args = args
        self.idx2user = dataffm['idx2user']
        self.idx2isbn = dataffm['idx2isbn']

        self.device = args.DEVICE

        self.mlp_dims = args.DCN_MLP_DIMS
        self.dropout = args.DCN_DROPOUT
        self.num_layers = args.DCN_NUM_LAYERS

        self.model = _FFDCNModel(self.ff_field_dims, self.ff_embed_dim, self.dcn_embed_dim, num_layers=self.num_layers, mlp_dims=self.mlp_dims, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.ff_train_dataloader, smoothing=0, mininterval=1.0)
            for i, (ff_fields, target) in enumerate(tk0):
                ff_fields, target = ff_fields.to(self.device), target.to(self.device)
                y = self.model(ff_fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
        #self.predict_train(True)


    def predict_train(self,save=False):
        self.model.eval()
        targets, predicts = list(), list()
        users, isbns = np.array([]),np.array([])
        with torch.no_grad():
            for ff_fields, target in tqdm.tqdm(self.ff_valid_dataloader, smoothing=0, mininterval=1.0):
                ff_fields, target = ff_fields.to(self.device), target.to(self.device)
                y = self.model(ff_fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                if save:
                    users = np.concatenate((users,ff_fields[:,0].tolist()))
                    isbns = np.concatenate((isbns,ff_fields[:,1].tolist()))
            if save:
                print(f'--------------- Saving Valid ---------------')
                df_valid = pd.DataFrame({
                    'user_id':users,
                    'isbn':isbns,
                    'target':targets,
                    'rating':predicts})
                df_valid['user_id'] = df_valid['user_id'].map(self.idx2user)
                df_valid['isbn'] = df_valid['isbn'].map(self.idx2isbn)
                df_valid.sort_values(by='user_id').to_csv(f'valid/valid_{self.args.MODEL}.csv',index=False)
        return rmse(targets, predicts)


    def predict(self, ff_dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for ff_fields in tqdm.tqdm(ff_dataloader, smoothing=0, mininterval=1.0):
                ff_fields = ff_fields[0].to(self.device)
                y = self.model(ff_fields)
                predicts.extend(y.tolist())
        return predicts