import csv
def dialogue_process(typename,index,reject,file_path_dia,tokenizer,len1=600,len2=800,len3=1200):
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
#                 if len(tokenizer(reject[i])['input_ids'])<150:
#                     len1=400
                reverse_lst = temp[::-1]
                flag=0
                for item in reverse_lst:
                    if flag==1:
                        if 'assessment' in item:
                            str_zhen=item+'\n'+str_zhen
                            continue
                        if '-year-old' not in item and 'year old' not in item:
                            continue
                        else:
                            str_zhen=item+'\n'+str_zhen
                            break
                    length=len(tokenizer(item+'\n'+str_zhen)['input_ids'])
                    if length<len1:
                        str_zhen=item+'\n'+str_zhen
                    else: 
                        if 'x-ray' in str_zhen or 'results' in str_zhen:
                            flag=1
                        if 'assessment' in item:
                            str_zhen=item+'\n'+str_zhen
                            continue
                        if 'physical examination' in item or 'physical exam' in item or 'vital' in item or 'review of systems' in item:
                            continue  
                        else:
                            if length<len2:
                                str_zhen=item+'\n'+str_zhen
                            else:
                                break
                    if length>len3:
                        break
                i=i+1
                dialogue.append(str_zhen)
#                 print(dialogue)
    return dialogue