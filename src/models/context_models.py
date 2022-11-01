import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss

import wandb

class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.args = args
        self.idx2user = data['idx2user']
        self.idx2isbn = data['idx2isbn']
        
        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        wandb.init(project='level1_bookratingprediction_recsys-level1-recsys-06', entity='thumbs-up')
        wandb.run.name = self.args.Wandb_name # 프로젝트 이름 설정부분, config 파일 맨 밑에 넣어놓음.
        wandb.config.update({
            "batch_size" : self.args.BATCH_SIZE,
            "epochs": self.args.EPOCHS,
            # "optimizer": self.args.optimizer
        })
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
            
            wandb.log({
            'rmse_score' : rmse_score
            # 'train_acc' : train_acc,
            # 'train_roc_auc' : train_roc_auc,
            # 'valid_loss' : valid_loss,
            # 'valid_acc' : valid_acc,
            # 'valid_roc_auc' : valid_roc_auc,
            })
        self.predict_train(True)


    def predict_train(self,save=False):
        self.model.eval()
        targets, predicts = list(), list()
        users, isbns = np.array([]),np.array([])
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                if save:
                    users = np.concatenate((users,fields[:,0].tolist()))
                    isbns = np.concatenate((isbns,fields[:,1].tolist()))
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


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.args = args
        self.idx2user = data['idx2user']
        self.idx2isbn = data['idx2isbn']

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
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
        self.predict_train(True)


    def predict_train(self,save=False):
        self.model.eval()
        targets, predicts = list(), list()
        users, isbns = np.array([]),np.array([])
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                if save:
                    users = np.concatenate((users,fields[:,0].tolist()))
                    isbns = np.concatenate((isbns,fields[:,1].tolist()))
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


    def predict(self, dataloader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts
