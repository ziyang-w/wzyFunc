'''
Descripttion: coding for deep learning: MLP, CNN, GNN, et al.
Author: ziyang-W, ziyangw@yeah.net
Co.: IMICAMS
Date: 2022-05-26 09:23:07
LastEditTime: 2022-05-26 11:24:06
Copyright (c) 2022 by ziyang-W (ziyangw@yeah.net), All Rights Reserved. 
'''
import os
import numpy as np
import datetime
import torch

class EarlyStopping(object):
    '''
    description: 用于深度学习模型中的早停
    return {*}

    REFERENCE:
        https://github.com/gu-yaowen/MODDA/utils.py
    '''
    def __init__(self, patience=10, saved_path='.'):
        dt = datetime.datetime.now()
        self.filename = os.path.join(saved_path, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
