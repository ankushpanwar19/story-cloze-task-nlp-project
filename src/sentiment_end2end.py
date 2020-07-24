import os
import argparse
import math
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import SentimentData
from all_models import SentimentNetEnd2End
from metrics import Accuracy
from sklearn.model_selection import train_test_split

import utils

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('running on gpu')
else:
    device=torch.device('cpu')
    print('running on cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=32)

FLAGS = parser.parse_args()
NUM_EPOCHS = FLAGS.num_epochs
LR = FLAGS.lr
BATCH_SIZE = FLAGS.batch_size
PRINT_EVERY = FLAGS.print_every


stories = utils.read_data('data/nlp2_val.csv')
test_stories = utils.read_data('data/nlp2_test.csv')

train_stories, val_stories = train_test_split(stories, test_size=0.2)

train_dataloader = DataLoader(SentimentData(train_stories,device,is_base_train=False), batch_size=BATCH_SIZE,
                                shuffle=True)
val_dataloader = DataLoader(SentimentData(val_stories,device, is_base_train=False), batch_size=BATCH_SIZE,
                                shuffle=False)
test_dataloader = DataLoader(SentimentData(test_stories,device, is_base_train=False), batch_size=BATCH_SIZE,
                                shuffle=False)

net = SentimentNetEnd2End(device, pretrained=True)
net.to(device)

ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

metric_acc = Accuracy()

n_iteration=len(train_dataloader)
v_iteration=len(val_dataloader)
val_accuracy_prev = 0

for epoch in range(NUM_EPOCHS):
    running_loss_train = 0.0
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        ending_sim = net(data)
        loss=ce_loss(ending_sim,data['labels'])
        loss.backward()
        optimizer.step()
        running_loss_train +=loss.item()

        if i%PRINT_EVERY == 0:
            print(f'Epoch: {epoch+1}, Step: {i}/{n_iteration},\
                Runningloss: {running_loss_train/PRINT_EVERY}')
            running_loss_train = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_dataloader):
            val_ending_sim = net(val_data)

            _, predicted = torch.max(val_ending_sim, 1)
            metric_acc.update_batch(predicted, val_data['labels'])
        val_accuracy = metric_acc.get_metrics_summary()
        metric_acc.reset()
        if val_accuracy > val_accuracy_prev:
            torch.save(net.state_dict(), 'checkpoints/sentiment_finetuned.pth')
            print('checkpoint saved')
            val_accuracy_prev = val_accuracy

        print(f'============Epoch: {epoch+1}, ValAccuracy: {val_accuracy}=================')

print('end')
