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

from thexp.utils import random
import torch.nn as nn
import torch
from torch.optim import SGD
stt = random.get_state()

data = torch.rand(5,2)
y = torch.tensor([0,0,0,0,0])
model = nn.Linear(2,2)
print(list(model.parameters()))

sgd = SGD(model.parameters(),lr=0.01)
logits = model(data)
loss = nn.CrossEntropyLoss()(logits,y)
loss.backward()
sgd.step()
sgd.zero_grad()

print(list(model.parameters()))

random.set_state(stt)

data = torch.rand(5,2)
y = torch.tensor([0,0,0,0,0])
model = nn.Linear(2,2)
print(list(model.parameters()))

sgd = SGD(model.parameters(),lr=0.01)
logits = model(data)
loss = nn.CrossEntropyLoss()(logits,y)
loss.backward()
sgd.step()
sgd.zero_grad()

print(list(model.parameters()))
