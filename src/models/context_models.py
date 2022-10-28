import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Dataset

import wandb


class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        # kfold에 의해서 필요 x
        # self.train_dataloader = data['train_dataloader']
        # self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.args = args
        
        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        # kfold에 사용될 새로 정의한
        self.seed = args.SEED
        self.trainx = data['train_X']
        self.trainy = data['train_y']
        self.batchsize = args.BATCH_SIZE

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        wandb.init()
        wandb.config.update({
            "batch_size" : self.args.BATCH_SIZE,
            "epochs": self.args.EPOCHS,
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
            # rmse 계산
            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation rmse:', rmse_score)
            
            wandb.log({
            'rmse_score' : rmse_score
            })

    def kfold_train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        wandb.init()
        wandb.config.update({
            "batch_size" : self.batchsize,
            "epochs": self.epochs,
        })
        train_dataset = TensorDataset(torch.LongTensor(self.trainx.values), torch.LongTensor(self.trainy.values))
        kfold = KFold(n_splits = 10, random_state = self.seed, shuffle = True)
        print(train_dataset)
        validation_loss = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, sampler=train_subsampler) # 해당하는 index 추출
            valloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, sampler=val_subsampler)
            
            for epoch in range(self.epochs):
                # model이 훈련하고 있음을 모델에 알림 -> 이는 훈련 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 계층에 정보 제공
                self.model.train()
                total_loss = 0
                tk0 = tqdm.tqdm(trainloader, smoothing=0, mininterval=1.0)
                
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
                              
            rmse_score = self.predict_train(valloader)
            rm = self.predict_train_real(trainloader)
            validation_loss.append(rmse_score)
            print('epoch:', epoch, 'validation rmse:', rmse_score)
            print("k-fold", fold,"  train rmse: %.4f" %(rm))
            wandb.log({
            'rmse_score' : rmse_score
            })
            # print(trainloader.shape)
            
        mean = np.mean(validation_loss)
        std = np.std(validation_loss)
        print("Validation Score: %.4f, ± %.4f" %(mean, std))
        
        
            
    # rmse 계산
    def predict_train(self, val):
        self.model.eval()
        targets, predicts = list(), list()
        
        with torch.no_grad():
            for fields, target in tqdm.tqdm(val, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)
    
    def predict_train_real(self, train):
        self.model.eval()
        targets, predicts = list(), list()
        
        with torch.no_grad():
            for fields, target in tqdm.tqdm(train, smoothing=0, mininterval=1.0):
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

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        for epoch in range(self.epochs):
            # model이 훈련하고 있음을 모델에 알림 -> 이는 훈련 및 평가 중에 다르게 동작하도록 설계된 Dropout 및 BatchNorm과 같은 계층에 정보 제공
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
