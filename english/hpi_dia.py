import csv
def dialogue_process(typename,index,reject,file_path_dia,tokenizer,len1=700,len2=1200):
    if typename=='test':
        encoding='gbk'
    else:
        encoding='utf8'
    with open(file_path_dia,encoding=encoding) as f:
            dialogue=[]
            f_csv=csv.reader(f)
            headers = next(f_csv)
            i=0
            count=0
            k=0          
            for row in f_csv:
#                 print(row[0])
                if row[0] not in index:
                    continue
                str1=row[2].replace('[doctor]','D:').replace('[patient]','P:')
                count=count+len(tokenizer(str1)['input_ids'])
                temp=str1.split('\n')
                str_zhen='' 
                if len(tokenizer(reject[i])['input_ids'])<150:
                    len1=400
                for item in temp:
                    length=len(tokenizer(str_zhen+item)['input_ids'])
                    if length<len1:
                        if 'examination' in item or 'vital' in item:
                            continue
                        str_zhen=str_zhen+item+'\n'
                        
                    elif length<len2:
                        if 'physical examination' in item or 'exam' in item or 'vital' in item or 'review of systems' in item or 'x-ray' in item or 'ct scan' in item:
#                             print(i)
#                             print('aaaa')
                            break
                        else:
                            str_zhen=str_zhen+item+'\n'
                    else:
                        break
                i=i+1
                dialogue.append(str_zhen)
#                 print(dialogue)
    return dialogue