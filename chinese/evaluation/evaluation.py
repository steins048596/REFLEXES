import json
import argparse
from rouge import Rouge
import csv
import re
import string
from zhon.hanzi import punctuation
import pkuseg
import csv
import jieba
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

file_path_zong='../data/test_index.csv'
file_path_cc='../new_experiment/cc_chatgpt_fewshot.csv'
file_path_hpi='../new_experiment/hpi_chatgpt_fewshot.csv'
file_path_sug='../new_experiment/sug_chatgpt_fewshot.csv'
file_path_zhenduan='../data/zhenduan.csv'
########################Rouge#####################################

def process(title, delimiter=''):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return delimiter.join(x)

def compute_rouge(source, targets):
    try:
        r1, r2, rl = 0, 0, 0
        n = len(targets)
        for target in targets:
            source, target = ' '.join(source), ' '.join(target)
            scores = Rouge().get_scores(hyps=source, refs=target)
            r1 += scores[0]['rouge-1']['f']
            r2 += scores[0]['rouge-2']['f']
            rl += scores[0]['rouge-l']['f']
        return {
            'rouge-1': r1 / n,
            'rouge-2': r2 / n,
            'rouge-l': rl / n,
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]
    results = {k: v / len(targets) for k, v in scores.items()}
#    print(results)
    # 在 CBLEU 评测中，我们使用 rouge-1, rouge-2 和 rouge-l 的平均值作为评价指标
    return (results['rouge-1'] + results['rouge-2'] + results['rouge-l']) / 3,results['rouge-1'],results['rouge-2'],results['rouge-l']


target1=[]
target2=[]
zhusu_text=[]
xianbingshi_text=[]
fuzhu_text=[]
jiwang_text=[]
zhenduan_text=[]
jianyi_text=[]

def my_split(s, ds):
    #  s1:待分隔字符串，ds:包含所有分隔符的字符串
    """
     需要注意有种情形是连续两个分隔符，如'i,,j'
     列表中会出现空字符串，此时就需要对结果进行过滤。
    """
    res = [s]
    for d in ds:
        t = []
        list(map(lambda x: t.extend(x.split(d)), res))
        res = t
    # 使用列表解析过滤空字符串
    return [x for x in res if x]


def pprocess(str1):
    split=my_split(str1,'、，')[:1]
    str1=('，').join(split)
    if str1[-1]!="。":
        str1=str1+"。"
    return str1


zhenduan_text=[]
with open(file_path_zong,encoding='gbk') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            fuzhu_text.append(row[10].replace('(3)辅助检查：',''))
            jiwang_text.append(row[13].replace('(4)既往史：',''))

zhusu_text=[]
xianbingshi_text=[]

jianyi_text=[]

with open(file_path_cc,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            zhusu_text.append(row[1].replace('(1)主诉：',''))

with open(file_path_hpi,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            xianbingshi_text.append(row[1].replace('(2)现病史：',''))

with open(file_path_zhenduan,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            zhenduan_text.append(pprocess(row[1].replace('(5)诊断：','')))

with open(file_path_sug,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            jianyi_text.append(row[1].replace('(6)建议：',''))     
target_zhen=[]

with open(file_path_zong,encoding='gbk') as f:
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if len(row[5])>len(row[6]):
                xianbingshi=row[5]
            else:
                xianbingshi=row[6]
            if len(row[17])>len(row[18]):
                jianyi=row[17]
            else:
                jianyi=row[18]
            str1='主诉：'+row[2]+'现病史：'+row[5]+'辅助检查：'+row[8]+'既往史：'+row[11]+'诊断：'+row[14]+'建议：'+row[17]
            str2='主诉：'+row[3]+'现病史：'+row[6]+'辅助检查：'+row[9]+'既往史：'+row[12]+'诊断：'+row[15]+'建议：'+row[18]
            str_zhen='主诉：'+row[3]+'现病史：'+xianbingshi+'辅助检查：'+row[9]+'既往史：'+row[12]+'诊断：'+row[15]+'建议：'+jianyi
            target1.append(str1)
            target2.append(str2)
            target_zhen.append(str_zhen)


for i in range(len(fuzhu_text)):
    if '无' in fuzhu_text[i] or '建议' in fuzhu_text[i] or '未' in fuzhu_text[i]:
        fuzhu_text[i]="暂缺。"
for i in range(len(jiwang_text)):
    if jiwang_text[i]!='无':
        if '无' in jiwang_text[i]:
            jiwang_text[i]="不详"
        if len(jiwang_text[i])>10:
            jiwang_text[i]="不详"

def save_txt(str3):    
    with open('test_eval.txt','a',encoding='utf-8') as f:
        f.write(str3+'\n')
    f.close()

golds, preds = [], []
ref=[]
print(len(zhusu_text))

for i in range(len(zhusu_text)):
    str1=target1[i]
    str2=target2[i]
    str3="主诉："+zhusu_text[i]+''+"现病史："+xianbingshi_text[i]+"辅助检查："+fuzhu_text[i]+"既往史："+jiwang_text[i]+"诊断："+zhenduan_text[i]+"建议："+jianyi_text[i]+"。"
    str3=str3.replace('\n','')
    
    if len(str1)>len(str2):
        str_zhen=str1
    else:
        str_zhen=str2
    ref.append(str_zhen)
    golds.append([str_zhen])
    preds.append(str3)
    save_txt(str3)
print('-- MRG task evaluation-rouge --')
print('rouge-avg/rouge-1/rouge-2/rouge-L')
print(compute_rouges(preds, golds))

########################Meteor#####################################
lexicon=[]
seg = pkuseg.pkuseg(user_dict=lexicon,model_name='medicine')
punc=string.punctuation
punc2=punctuation
def seg_depart(sentence):
    sentence_depart = seg.cut(sentence.replace('\u3000', '').replace(' ',''))
    temp=[]
    for word in sentence_depart:
        if word not in punc and word not in punc2:
            if word != '\t' and word != '\r' and word != '\n':
                temp.append(word)
    return temp
input_list=[]
input_list2=[]
refer_list=[]
j=0
for i in range(len(preds)):
    input_list.append(seg_depart(preds[i]))
    refer_list.append([])
    refer_list[j].append(seg_depart(ref[i]))
    j=j+1
ini=0

for i in range(len(input_list)):
    score = meteor_score(refer_list[i], input_list[i])  
    ini=ini+score
print('meteor:',ini/len(input_list))