"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
import sys
import time
sys.path.insert(0, "../")
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torchvision.datasets import FakeData

from thexp.frame.meter import Meter
from thexp.frame.params import Params
from thexp.frame.trainer import Trainer
from thexp.utils.optim import OptimParam
from torchvision import transforms
# from torch.utils.data.dataloader import DataLoader
from thexp.utils.date.dataloader import DataLoader

train = FakeData(image_size=(1,28,28),transform=transforms.ToTensor())
eval = FakeData(image_size=(1,28,28),transform=transforms.ToTensor())
test = FakeData(image_size=(1,28,28),transform=transforms.ToTensor())

train_loader = DataLoader(train, shuffle=True, batch_size=32,drop_last=True)
eval_loader = DataLoader(eval, shuffle=True, batch_size=32)
test_loader = DataLoader(test, shuffle=True, batch_size=32)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


cross = nn.CrossEntropyLoss()


class MyTrainer(Trainer):
    def __init__(self, params: Params):
        super().__init__(params)
        self.model = MyModel()
        self.optim = OptimParam.build().SGD().finish().create(OptimParam.O_SGD, self.model.parameters())
        self.regist_databundler(
            train=train_loader,
            test=test_loader,
            eval=eval_loader,
        )

    def train_epoch(self, eidx, params):
        super().train_epoch(eidx, params)

    def train_batch(self, eidx, idx, global_step, batch_data, params, device):
        self.iter_train_dataloader().set_batch_size(eidx + idx//10 + 5)
        optim = self.optim
        meter = Meter()
        xs, ys = batch_data
        logits = self.model(xs)
        meter.eidx = eidx
        meter.loss1 = cross(logits, ys)
        meter.loss2 = 1
        meter.shape = xs.shape
        meter.lens = ys.shape
        # meter.loss5 = cross(logits, ys)
        # meter.loss6 += cross(logits, ys)
        # meter.loss6 += cross(logits, ys)
        meter.loss1.backward()
        optim.step()
        optim.zero_grad()
        return meter


params = Params()
params.build_exp_name("mytrainer")
trainer = MyTrainer(params)
trainer.initial_exp("./experiment")
trainer.train()
