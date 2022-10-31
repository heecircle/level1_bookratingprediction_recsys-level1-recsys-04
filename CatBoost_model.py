import os
import pandas as pd
import numpy as np
import catboost

from catboost import CatBoostRegressor, Pool, metrics, cv
from sklearn.metrics import accuracy_score
from ._models import rmse, RMSELoss

class CatBoost:

    def __init__(self, args, data):
        super().__init__()
        # wandb.config.update(args)
        self.criterion = RMSELoss()

        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_valid = data['X_valid']
        self.y_valid = data['y_valid']
        self.sub = data['sub']
        self.cat_features = list(range(0, self.X_train.shape[1]))

        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.seed = args.SEED

        self.model = CatBoostRegressor(iterations=self.epochs, loss_function='RMSE', random_seed=self.seed, learning_rate=self.learning_rate, 
            verbose=20)


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        self.model.fit(
            self.X_train, self.y_train,
            cat_features=self.cat_features,
            eval_set=(self.X_valid, self.y_valid),
        )

        # for epoch in range(self.epochs):
        #     self.model.fit(self.X_train, y=self.y_train, cat_features=self.cat_features, eval_set=(self.X_valid,self.y_valid))
        #     total_loss = 0
        #     tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
        #     # for i, (fields, target) in enumerate(tk0):
        #     #     y = self.model(fields)
        #     #     loss = self.criterion(y, target.float())
        #     #     self.model.zero_grad()
        #     #     loss.backward()
        #     #     self.optimizer.step()
        #     #     total_loss += loss.item()
        #     #     if (i + 1) % self.log_interval == 0:
        #     #         tk0.set_postfix(loss=total_loss / self.log_interval)
        #     #         total_loss = 0

        #     rmse_score = self.predict_train()
        #     # wandb.log({
        #     #     "RMSE": rmse_score,
        #     #     "Loss": total_loss
        #     #     })
        #     print('epoch:', epoch, 'validation: rmse:', rmse_score)


    # def predict_train(self):
    #     self.model.eval()
    #     targets, predicts = list(), list()
    #     with torch.no_grad():
    #         for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
    #             y = self.model(fields)
    #             targets.extend(target.tolist())
    #             predicts.extend(y.tolist())
    #     return rmse(targets, predicts)


    def predict(self):
        predicts = self.model.predict(self.sub)
        return predicts