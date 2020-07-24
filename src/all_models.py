import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

import utils


class BertNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.out = torch.nn.Linear(768, 1)

        self.device = device

    def forward(self, data, return_emb_sim=False):
        ending1, ending2 = self.preprocess_input(data)

        e1 = self.get_embedding(ending1)
        e1_score = self.out(e1)

        e2 = self.get_embedding(ending2)
        e2_score = self.out(e2)

        output = torch.cat((e1_score, e2_score), dim=1)
        if not return_emb_sim:
            return output
        
        cosine_sim = F.cosine_similarity(e1, e1, dim=1)
        # emb_output = torch.stack((e1, e2), dim=-1)
        return output, cosine_sim

    def preprocess_input(self, batch):
        e1_inputs = self.tokenizer(text=batch['full_story'], 
                        text_pair=batch['ending1'],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",)
        e2_inputs = self.tokenizer(text=batch['full_story'], 
                        text_pair=batch['ending2'],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt")

        for i in e1_inputs.keys():
            e1_inputs[i]=e1_inputs[i].to(self.device)
            e2_inputs[i]=e2_inputs[i].to(self.device)

        return e1_inputs, e2_inputs

    def get_embedding(self, ending):
        e1 = self.bert(**ending)
        return e1[0][:,0,:]


class SentimentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(3,256,batch_first=True)
        self.out = torch.nn.Linear(256, 3)

    def forward(self,data, return_emb=False):
        _,(h,_)=self.lstm1(data)
        output = self.out(h[-1])
        if not return_emb:
            return output
        
        return output, h[-1]

class SentimentNetEnd2End(torch.nn.Module):
    def __init__(self, device, pretrained):
        super().__init__()
        # self.senti_net = SentimentNet()
        self.senti_net = self._create(device, pretrained=pretrained)
        self.criterion=torch.nn.CosineSimilarity()

    def _create(self, device, pretrained=True):
        model = SentimentNet()
        if pretrained:
            state_dict = torch.load("checkpoints/sentiment_base.pth", map_location=device)
            model.load_state_dict(state_dict)
            print('Checkpoint loaded')

        return model

    def forward(self, data, return_emb_sim=False):
        # output,(h,c)=self.lstm1(data['story_senti_emb'])
        senti_pred = self.senti_net(data['story_senti_emb'])

        ending1_sim = self.criterion(senti_pred,data['ending1_senti_emb'])
        ending2_sim = self.criterion(senti_pred,data['ending2_senti_emb'])

        ending_sim = torch.stack((ending1_sim, ending2_sim), dim=1)

        if not return_emb_sim:
            return ending_sim
        
        cosine_sim = F.cosine_similarity(data['ending1_senti_emb'], data['ending2_senti_emb'], dim=1)
        return ending_sim, cosine_sim

class CommonsenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = torch.nn.Linear(4, 256)
        self.out = torch.nn.Linear(256, 1)

    def forward(self, data, return_emb_sim=False):
        out1=self.step(data['ending1_common_sense'])
        out2=self.step(data['ending2_common_sense'])

        ending_prob = torch.cat((out1, out2),dim=1)

        if not return_emb_sim:
            return ending_prob

        cosine_sim = F.cosine_similarity(data['ending1_common_sense'], data['ending2_common_sense'], dim=1)
        return ending_prob, cosine_sim

    def step(self,ending):
        out=F.relu(self.layer1(ending))
        out=self.out(out)

        return out

class CombinedNet(torch.nn.Module):
    def __init__(self, device, pretrained=(True, True, True)):
        super().__init__()

        self.bert_net = self._create(device, BertNet(device), 
                                    ckpt_path="checkpoints/bert.pth", pretrained=pretrained[0])
        self.sentiment_net = self._create(device, SentimentNetEnd2End(device, pretrained=False), 
                                    ckpt_path="checkpoints/sentiment_finetuned.pth", pretrained=pretrained[1])
        self.commonsense_net = self._create(device, CommonsenseNet(), 
                                    ckpt_path="checkpoints/common_sense.pth", pretrained=pretrained[2])

        self.device = device
        self.gate = torch.nn.Linear(3,3)
        # self.gates = torch.nn.Sequential(
        #                         torch.nn.Linear(6, 3),
        #                         torch.nn.Softmax(dim=1))       

    def _create(self, device, model_cls, ckpt_path, pretrained=True):
        model = model_cls
        if pretrained:
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f'{ckpt_path} loaded')

        return model


    def forward(self, data):
        bert_logits, bert_sim = self.bert_net(data, return_emb_sim=True)
        senti_logits, senti_sim = self.sentiment_net(data, return_emb_sim=True)
        commonsense_logits, commonsense_sim = self.commonsense_net(data, return_emb_sim=True)

        combined_sim = torch.stack((bert_sim, senti_sim, commonsense_sim), dim=1)
        gates = F.softmax(self.gate(combined_sim), dim=1) 
        gates = torch.unsqueeze(gates, dim=-1)

        combined_logits = torch.stack((bert_logits, senti_logits, commonsense_logits), dim=1)
        combined_probs = F.softmax(combined_logits, dim=-1)

        final_probs = (gates * combined_probs).sum(dim=1)

        return final_probs

