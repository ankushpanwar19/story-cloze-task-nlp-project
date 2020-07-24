import os
import argparse
import math
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle

from dataset import CombinedData
from all_models import CombinedNet
from metrics import Accuracy

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
stories_test = utils.read_data('data/nlp2_test.csv')

embed_file_val = open("data/dictionary_commonsense_val.pickle",'rb')
embedding_val=pickle.load(embed_file_val)
embed_file_val.close()

embed_file_test = open("data/dictionary_commonsense_test.pickle",'rb')
embedding_test=pickle.load(embed_file_test)
embed_file_test.close()


train_stories, val_stories = train_test_split(stories, test_size=0.1)


train_dataloader = DataLoader(CombinedData(train_stories, embedding_val, device), batch_size=BATCH_SIZE,
                                shuffle=True)
val_dataloader = DataLoader(CombinedData(val_stories, embedding_val, device), batch_size=BATCH_SIZE,
                                shuffle=False)
test_dataloader = DataLoader(CombinedData(stories_test, embedding_test, device), batch_size=BATCH_SIZE,
                                shuffle=False)

net = CombinedNet(device)
net.to(device)

ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

metric_acc = Accuracy()

n_iteration = len(train_dataloader)
v_iteration = len(val_dataloader)
val_accuracy_prev = 0

for epoch in range(NUM_EPOCHS):
    running_loss_train = 0.0
    running_loss_val = 0.0
    for i, train_batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        logits = net(train_batch)
        train_loss = ce_loss(logits, train_batch['labels'])
        # output, train_loss = utils.run_step(train_batch, net, tokenizer, ce_loss, device)

        train_loss.backward()
        optimizer.step()

        running_loss_train += train_loss.item()

        _, predicted = torch.max(logits, 1)
        metric_acc.update_batch(predicted, train_batch['labels'])

        if i%PRINT_EVERY == 0:
            train_accuracy = metric_acc.get_metrics_summary()
            metric_acc.reset()

            print(f'Epoch: {epoch+1}, Step: {i}/{n_iteration}, Accuracy: {train_accuracy}, \
                    Runningloss: {running_loss_train/PRINT_EVERY}')
            running_loss_train = 0


    with torch.no_grad():
        for i, val_batch in enumerate(val_dataloader):
            logits = net(val_batch)
            val_loss = ce_loss(logits, val_batch['labels'])
            # logits, val_loss = utils.run_step(val_batch, net, tokenizer, ce_loss, device)

            running_loss_val += val_loss.item()

            _, predicted = torch.max(logits, 1)
            metric_acc.update_batch(predicted, val_batch['labels'])

        val_accuracy = metric_acc.get_metrics_summary()
        metric_acc.reset()

        if val_accuracy > val_accuracy_prev:
            torch.save(net.state_dict(), 'checkpoints/combined_model.pth')
            print('checkpoint saved')
            val_accuracy_prev = val_accuracy

        print(f'============Epoch: {epoch+1}, ValAccuracy: {val_accuracy}, Valloss: {running_loss_val/v_iteration}=================')