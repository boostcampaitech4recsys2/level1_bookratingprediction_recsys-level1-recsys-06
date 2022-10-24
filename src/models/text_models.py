import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from ._models import RMSELoss, FeaturesEmbedding, FactorizationMachine_v


class CNN_1D(nn.Module):
    def __init__(self, word_dim, out_dim, kernel_size, conv_1d_out_dim):
        super(CNN_1D, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv1d(
                                        in_channels=word_dim,
                                        out_channels=out_dim,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(kernel_size, 1)),
                                nn.Dropout(p=0.5)
                                )
        self.linear = nn.Sequential(
                                    nn.Linear(int(out_dim/kernel_size), conv_1d_out_dim),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5))

    def forward(self, vec):
        output = self.conv(vec)
        output = self.linear(output.reshape(-1, output.size(1)))
        return output


class _DeepCoNN(nn.Module):
    def __init__(self, field_dims, embed_dim, word_dim, out_dim, kernel_size, conv_1d_out_dim, latent_dim):
        super(_DeepCoNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cnn_u = CNN_1D(
                             word_dim=word_dim,
                             out_dim=out_dim,
                             kernel_size=kernel_size,
                             conv_1d_out_dim=conv_1d_out_dim,
                            )
        self.cnn_i = CNN_1D(
                             word_dim=word_dim,
                             out_dim=out_dim,
                             kernel_size=kernel_size,
                             conv_1d_out_dim=conv_1d_out_dim,
                            )
        self.fm = FactorizationMachine_v(
                                         input_dim=(conv_1d_out_dim * 2) + (embed_dim*len(field_dims)),
                                         latent_dim=latent_dim,
                                         )
    def forward(self, x):
        user_isbn_vector, user_text_vector, item_text_vector = x[0], x[1], x[2]
        user_isbn_feature = self.embedding(user_isbn_vector)
        user_text_feature = self.cnn_u(user_text_vector)
        item_text_feature = self.cnn_i(item_text_vector)
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    user_text_feature,
                                    item_text_feature
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)


class DeepCoNN:
    def __init__(self, args, data):
        super(DeepCoNN, self).__init__()
        self.device = args.DEVICE
        self.model = _DeepCoNN(
                                np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32),
                                args.DEEPCONN_EMBED_DIM,
                                args.DEEPCONN_WORD_DIM,
                                args.DEEPCONN_OUT_DIM,
                                args.DEEPCONN_KERNEL_SIZE,
                                args.DEEPCONN_CONV_1D_OUT_DIM,
                                args.DEEPCONN_LATENT_DIM
                                ).to(self.device)
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.train_data_loader = data['train_dataloader']
        self.valid_data_loader = data['valid_dataloader']
        self.criterion = RMSELoss()
        self.epochs = args.EPOCHS
        self.model_name = 'text_model'


    def train(self):
        minimum_loss = 999999999
        loss_list = []
        tk0 = tqdm.tqdm(range(self.epochs), smoothing=0, mininterval=1.0)
        for epoch in tk0:
            self.model.train()
            total_loss = 0
            n = 0
            for i, data in enumerate(self.train_data_loader):
                if len(data)==3:
                    fields, target = [data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
                elif len(data)==4:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
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
                if len(data)==3:
                    fields, target = [data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
                elif len(data)==4:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
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
                if len(data)==3:
                    fields, target = [data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
                elif len(data)==4:
                    fields, target = [data['user_isbn_vector'].to(self.device), data['user_summary_merge_vector'].to(self.device), data['item_summary_vector'].to(self.device)], data['label'].to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return predicts
