import paddle
from paddle import nn
import paddle.nn.functional as F
paddle.device.set_device('gpu:3')
import os
from paddlenlp.datasets import load_dataset
import paddlenlp
from paddlenlp.transformers import ErniePretrainedModel
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.data import Stack, Dict, Pad
import json
import csv
from functools import partial
import numpy as np
from paddle.io import DataLoader, BatchSampler



def read(filename,val=False):
    reject=[]
    accept=[]
    count=0
    minmin=[]
    with open(filename,encoding='gbk') as f:
        f_csv=csv.reader(f)
        headers = next(f_csv)
        i=0
        for row in f_csv:
            if len(str(row[18]))>=len(str(row[17])):
                acceptm=str(row[18])
            else:
                acceptm=str(row[17])
            accept_text=acceptm
            
            if str(row[19]).replace('(6)建议：','')==str(row[17]) or str(row[19]).replace('(6)建议：','')==str(row[18]):
                continue
            reject_text=row[19].replace('(6)建议：','').replace('(6) 建议：','')
            if len(reject_text)-len(accept_text)>50:
                continue
            if count>=60 and len(reject_text)>len(accept_text):
                continue
            reject.append(reject_text)
            accept.append(accept_text.replace('。','，'))
            
            if len(reject_text)>len(accept_text):
                count=count+1
                mina=len(reject_text)-len(accept_text)
                minmin.append(mina)
            i=i+1
            if val==False and i==80:
                break
            if val==True and i==10:
                break
    f.close()
    print(len(reject))
    print(len(accept))
    for i in range(len(accept)):
        yield {
            'reject': reject[i],
            'accept': accept[i]
        }
train_ds = load_dataset(read, filename='../data/train_index.csv', val=False,lazy=False)
dev_ds = load_dataset(read, filename='../data/dev_index.csv',val=True, lazy=False)




# 设置模型名称
MODEL_NAME2='ernie-health-chinese'
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained(MODEL_NAME2)


def preprocess_function(examples, tokenizer, max_seq_length=512):
    result={}
    result = tokenizer(text=examples["accept"], max_seq_len=max_seq_length)
    result['reject_ids'] = tokenizer(text=examples["reject"], max_seq_len=max_seq_length)['input_ids']
    result['chosen_mask'] = np.ones(len(result['input_ids']),dtype='int64')
    result['reject_mask'] = np.ones(len(result['reject_ids']),dtype='int64')
    return result

batchsize = 10

max_seq_length =512

train_trans_func = partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

dev_trans_func = partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)

test_trans_func = partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)
train_ds.map(train_trans_func, lazy=False)   
dev_ds.map(dev_trans_func, lazy=False)
#test_ds.map(test_trans_func, lazy=False)



# collate_fn函数构造，将不同长度序列充到批中数据的最大长度，再将数据堆叠
batchify_fn = lambda samples, fn=Dict({
    "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id), 
    "reject_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    "chosen_mask": Pad(axis=0,pad_val=0, dtype="int64"),
    "reject_mask": Pad(axis=0, pad_val=0, dtype="int64"),
}): fn(samples)
# 定义BatchSampler，选择批大小和是否随机乱序，进行DataLoader
train_batch_sampler = BatchSampler(train_ds, batch_size=batchsize, shuffle=True)
dev_batch_sampler = BatchSampler(dev_ds, batch_size=batchsize, shuffle=False)
train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=batchify_fn,return_list=True)
dev_data_loader = DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=batchify_fn,return_list=True)



class ValueErnie(ErniePretrainedModel):
    def __init__(self, config,dropout=None):
        super(ValueErnie, self).__init__(config)        
        self.ernie = ErnieModel(config)
        self.value_head = nn.Linear(self.ernie.config['hidden_size'], 1)
        self.dropout=nn.Dropout(dropout if dropout is not None else
                                  self.ernie.config["hidden_dropout_prob"])

    def forward(self, chosen_ids, chosen_mask):
        chosen_ids=chosen_ids.squeeze(1)
        chosen = self.ernie(chosen_ids,attention_mask=chosen_mask)
        chosen_output = self.dropout(chosen[0])
        chosen_logits = self.value_head(chosen_output)
        chosen_reward=chosen_logits.mean(axis=1).squeeze(1)
        return chosen_reward

from paddlenlp.transformers import ErnieModel, ErnieTokenizer
model = ValueErnie.from_pretrained('real_model/ernie-health-chinese/',dropout=0.1)

def loss_fn(logits):
    probs = F.log_sigmoid(logits)
    loss = -probs.mean()
    return loss



learning_rate = 5e-6

# 训练轮次
epochs = 6
# 学习率预热比例
warmup_proportion = 0.0

# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.0

max_grad_norm = 1.0

num_training_steps = len(train_data_loader) * epochs

# 学习率衰减策略
lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(learning_rate, num_training_steps,
                                    warmup_proportion)

decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params,
    grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm))

@paddle.no_grad()
def evaluation(model, data_loader):
    model.eval()    
    dist = 0
    on = 0
    cnt = 0
    for batch in data_loader:
        input_ids,reject_ids,chosen_mask,reject_mask= batch
        chosen_reward= model(input_ids,chosen_mask)
        reject_reward=model(reject_ids,reject_mask)
        for i in range(len(chosen_reward)):
            cnt += 1
            if chosen_reward[i] > reject_reward[i]:
                on += 1
        dist += (chosen_reward - reject_reward).mean().item()
    dist_mean = dist / len(data_loader)
    acc = on / cnt
    model.train()

    print('Total samples: %d' % len(data_loader.dataset))
    print('Slot dist_mean: %.4f' % dist_mean)
    print('Intent acc: %.4f\n' % acc)
    return dist_mean, acc

global_step = 0
best_dist_score = 0
ckpt_dir = "real_model/"
for epoch in range(1, epochs + 1):
#for epoch in range(1, 3):
    for step, batch in enumerate(train_data_loader, start=1):
        global_step += 1
        input_ids,reject_ids,chosen_mask,reject_mask= batch
        chosen_reward= model(input_ids,chosen_mask)
        reject_reward= model(reject_ids,reject_mask)
        logits =chosen_reward-reject_reward
        loss = loss_fn(logits)
        pp0=logits
        pp=loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    print("epoch: %d, batch: %d, loss: %.5f" % (epoch, step, loss))
    print('\nEval begin...')
    save_dir = ckpt_dir
    dist, acc  = evaluation(model, dev_data_loader)
    if dist > best_dist_score:
        best_dist_score = dist
        print("tempepoch:",epoch)
        print(dist)
        print(acc)
        paddle.save(model.state_dict(), save_dir+"few_shot_try/jianyi.pdparams")
        paddle.save(optimizer.state_dict(), save_dir+"few_shot_try/jianyi.pdopt")
#save_dir = ckpt_dir
#model.save_pretrained(save_dir)
#tokenizer.save_pretrained(save_dir)