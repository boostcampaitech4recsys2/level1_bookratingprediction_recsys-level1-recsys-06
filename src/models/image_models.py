import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from ._models import RMSELoss, FeaturesEmbedding, FactorizationMachine_v


class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()
        self.cnn_layer = nn.Sequential(
                                        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(6),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(12),
                                        nn.ReLU(),
                                        )
    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 12 * 4 * 4)
        return x


class _CNN_FM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, latent_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cnn = CNN_Base()
        self.fm = FactorizationMachine_v(
                                         input_dim=(embed_dim * 2) + (12 * 4 * 4),
                                         latent_dim=latent_dim,
                                         )

    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)
        img_feature = self.cnn(img_vector)
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)


class CNN_FM:
    def __init__(self, args, data):
        super().__init__()
        self.device = args.DEVICE
        self.model = _CNN_FM(
                            np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32),
                            args.CNN_FM_EMBED_DIM,
                            args.CNN_FM_LATENT_DIM
                            ).to(self.device)
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.train_data_loader = data['train_dataloader']
        self.valid_data_loader = data['valid_dataloader']
        self.criterion = RMSELoss()
        self.epochs = args.EPOCHS
        self.model_name = 'image_model'


    def train(self):
        minimum_loss = 999999999
        loss_list = []
        tk0 = tqdm.tqdm(range(self.epochs), smoothing=0, mininterval=1.0)
        for epoch in tk0:
            self.model.train()
            total_loss = 0
            n = 0
            for i, data in enumerate(self.train_data_loader):
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1
            self.model.eval()
            val_total_loss = 0
            val_n = 0
            for i, data in enumerate(self.valid_data_loader):
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                val_total_loss += loss.item()
                val_n += 1
            if minimum_loss > (val_total_loss/val_n):
                minimum_loss = (val_total_loss/val_n)
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{}.pt'.format(self.model_name))
                loss_list.append([epoch, total_loss/n, val_total_loss/val_n, 'Model saved'])
            else:
                loss_list.append([epoch, total_loss/n, val_total_loss/val_n, 'None'])
            tk0.set_postfix(train_loss=total_loss/n, valid_loss=val_total_loss/val_n)


    def predict(self, test_data_loader):
        self.model.eval()
        self.model.load_state_dict(torch.load('./models/{}.pt'.format(self.model_name)))
        targets, predicts = list(), list()
        with torch.no_grad():
            for data in test_data_loader:
                if len(data) == 2:
                    fields, target = [data['user_isbn_vector'].to(self.device)], data['label'].to(self.device)
                else:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['img_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return predicts
