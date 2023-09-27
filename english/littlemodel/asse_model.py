import torch
import torch.nn as nn
from transformers import AutoTokenizer, LongformerModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import os
import csv
import json
from apdia import dialogue_process
from transformers.optimization import Adafactor 
from torch.utils.data import Dataset
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")
    
class LEDRM(nn.Module):
    """
    BLOOM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BloomConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: str = None,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.model =LongformerModel.from_pretrained('../longformer_clinical')
        self.value_head = nn.Linear(self.model.config.hidden_size, 1, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))
    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value =values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value
    
tokenizer =AutoTokenizer.from_pretrained('../longformer_clinical')
tokenizer.pad_token = tokenizer.eos_token


model = LEDRM().to(dev)


train_data=[]
valid_data=[]
test_data=[]
def read_data(typename,file_path_chosen,file_path_reject,file_path_dia):   
    index=[]
    chosen=[]
    reject=[]
    with open(file_path_reject, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(file_path_chosen, 'r', encoding='utf8') as f:
        data2 = json.load(f)
    for i in range(len(data['data'])):
        if data['data'][i]["asessment_and_plan"]["ASSESSMENT AND PLAN"]!='' and data2['data'][i]["asessment_and_plan"]["ASSESSMENT AND PLAN"]!='':
            index.append(data['data'][i]['encounter_id'])
    for item in data['data']:
        if item['encounter_id'] not in index:
            continue
        else:
            reject.append(item["asessment_and_plan"]["ASSESSMENT AND PLAN"])
    for item in data2['data']:
        if item['file'][:item['file'].rfind('-')].replace('-','') in index:
            chosen.append(item["asessment_and_plan"]["ASSESSMENT AND PLAN"])
        else:               
            continue
    dialogue=dialogue_process(typename,index,reject,file_path_dia,tokenizer,300,550,1300)
    print(len(index))
    for i in range(len(index)):
        temp={'index':index[i],'prompt':dialogue[i],'chosen':chosen[i],'rejected':reject[i].replace('? ',' ')}
        if typename=='train':
            train_data.append(temp)
        if typename=='val':
            valid_data.append(temp)
        if typename=='test':
            test_data.append(temp)
read_data('train','../data/assessment/train.json','../data/train_ini_real.json','../data/assessment/train_apdia.csv')
read_data('test','../data/assessment/test.json','../data/assessment/test_ini.json','../data/assessment/test_apdia_zhen.csv')
read_data('val','../data/assessment/valid.json','../data/valid_ini_real.json','../data/assessment/valid_apdia.csv')

for data in valid_data:
    if data['index']=='D2N086':
        print(len(tokenizer(data['prompt'] + data['chosen'] + tokenizer.eos_token)['input_ids']))

class RmStaticDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer, max_length: int, special_token=None) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        max_token=0
        for data in tqdm(dataset):
            prompt = data['prompt']
            chosen = prompt + data['chosen'] + self.end_token
            if len(tokenizer(chosen)['input_ids'])>max_token:
                max_token=len(tokenizer(chosen)['input_ids'])
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            if len(chosen_token['input_ids'])>max_token:
                max_token=len(chosen_token['input_ids'])
                print(data['index'])
            self.chosen.append({
                "input_ids": chosen_token['input_ids'],
                "attention_mask": chosen_token['attention_mask']
            })
            
            reject = prompt + data['rejected'] + self.end_token
            if len(tokenizer(reject)['input_ids'])>max_token:
                max_token=len(tokenizer(reject)['input_ids'])
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'],
                "attention_mask": reject_token['attention_mask']
            })
        print('max_token',max_token)
    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx][
            "input_ids"], self.reject[idx]["attention_mask"]

def loss_fn(chosen_reward: torch.Tensor, reject_reward: torch.Tensor):
    probs = torch.sigmoid(chosen_reward - reject_reward)
    log_probs = torch.log(probs)
    loss = -log_probs.mean()
    return loss

train_dataset = RmStaticDataset(train_data, tokenizer, 1400)
valid_dataset = RmStaticDataset(valid_data, tokenizer, 1400)
test_dataset = RmStaticDataset(test_data, tokenizer, 1400)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              sampler=None,
                              batch_size=6,
                              pin_memory=True)

valid_dataloader = DataLoader(valid_dataset,
                              shuffle=True,
                              sampler=None,
                              batch_size=6,
                              pin_memory=True)

test_dataloader = DataLoader(test_dataset,
                             shuffle=True,
                             sampler=None,
                             batch_size=6,
                             pin_memory=True)



optimizer = Adafactor(
    model.parameters(),
    lr=5e-6,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

def evaluation(model, dataloader):
    model.eval()
    dist, on, cnt = 0, 0, 0
    with torch.no_grad():
        for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
            chosen_ids = chosen_ids.squeeze(1).to(dev)
            c_mask = c_mask.squeeze(1).to(dev)
            reject_ids = reject_ids.squeeze(1).to(dev)
            r_mask = r_mask.squeeze(1).to(dev)
            chosen_reward = model(chosen_ids, attention_mask=c_mask)
            reject_reward = model(reject_ids, attention_mask=r_mask)
            for i in range(len(chosen_reward)):
                cnt += 1
                if chosen_reward[i] > reject_reward[i]:
                    on += 1
            dist += (chosen_reward - reject_reward).mean().item()
        dist_mean = dist / len(dataloader)
        acc_total = on / cnt
    model.train()

    print('Total samples: %d' % len(dataloader.dataset))
    print('Slot dist_mean: %.4f' % dist_mean)
    print('Intent acc: %.4f\n' % acc_total)
    return dist_mean, acc_total

epochs=16
best_dist_score = 0
for epoch in range(1, epochs + 1):
    cnt = 0
    for chosen_ids, c_mask, reject_ids, r_mask in train_dataloader:
        chosen_ids = chosen_ids.squeeze(1).to(dev)      
        c_mask = c_mask.squeeze(1).to(dev)
        reject_ids = reject_ids.squeeze(1).to(dev)
        r_mask = r_mask.squeeze(1).to(dev)
        chosen_reward = model(chosen_ids, attention_mask=c_mask)
        reject_reward = model(reject_ids, attention_mask=r_mask)
        loss = loss_fn(chosen_reward, reject_reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
    print("epoch: %d, loss: %.5f" % (epoch, loss))
    print('\nEval begin...')
    dist, acc  = evaluation(model, valid_dataloader)
    if acc >=best_dist_score:
        best_dist_score = acc
        print("tempepoch:",epoch)
        print(dist)
        print(acc)
        torch.save(model.state_dict(), "ass_try.pdparams")