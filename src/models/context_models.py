import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss

from torch.utils.data import TensorDataset

from sklearn.model_selection import KFold

import pandas as pd

import os


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

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
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


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        # self.train_dataloader = data['train_dataloader']
        # self.valid_dataloader = data['valid_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)
        
        self.seed = args.SEED
        self.batchsize = args.BATCH_SIZE
        self.data_path = args.DATA_PATH
        self.model_name = args.MODEL
        
        
        self.trainx = data['train_X']
        self.trainy = data['train_y']

    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        minimum_loss = 999999999
        patience_limit = 3
        patience_check = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            cnt = 0
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0): # i: step
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += (loss.item() ** 2) * fields.shape[0]
                cnt += fields.shape[0]
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=np.sqrt(total_loss / cnt))

            rmse_score = self.predict_train()
            print('epoch:', epoch, 'validation: rmse:', rmse_score)

            if minimum_loss > rmse_score:
                patience_check = 0
                minimum_loss = rmse_score
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(self.model.state_dict(), './models/{}.pt'.format(self.model_name))
                print('updated!')
            else:
                patience_check += 1
                if patience_check >= patience_limit:
                    break
                print(f'patience_check: {patience_check}')
        return minimum_loss


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


    def kfold_train(self, data):
        train_dataset = TensorDataset(torch.LongTensor(data['train'].drop('rating', axis=1).values), torch.LongTensor(data['train']['rating'].values))
        kfold = KFold(n_splits = 10, random_state = self.seed, shuffle = True)

        validation_loss = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, sampler=train_subsampler) # sampindex 추출
            self.valid_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, sampler=val_subsampler)
            
            self.train()
            
            # 저장하는 부분
            predicts = self.predict(data['test_dataloader'])
            submission = pd.read_csv(self.data_path + 'sample_submission.csv')
            submission['rating'] = predicts
            submission.to_csv('submit/{}_{}.csv'.format(self.model_name, fold), index=False)

    def predict(self, dataloader):
        self.model.eval()
        self.model.load_state_dict(torch.load('./models/{}.pt'.format(self.model_name)))
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        return predicts