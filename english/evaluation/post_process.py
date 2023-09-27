import csv
file_path_ass='../results/asse.csv'
file_path_hpi='../results/hpi.csv'
hpi_refine=[]
index_zhen=[]
ours={}
test_ini={}
note=[]

with open(file_path_ass,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            index_zhen.append(row[0])
            hpi_refine.append(row[1])
            ours.setdefault(row[0],row[1])        

with open('../data/assessment/test_ini.csv',encoding='utf8') as f:
        f_csv=csv.reader(f)
        headers = next(f_csv)
        headers = next(f_csv)
        for row in f_csv:
            note.append(row[2])
            test_ini.setdefault(row[0],row[2])
            
lack=[]
lack_c=[]
for key,value in test_ini.items():
    if 'ASSESSMENT AND PLAN' in value and key in index_zhen:
        print('TRUE')
    elif key in index_zhen:
        lack.append(key)
        lack_c.append(value)
        
fenduan=[]
for item in index_zhen:
    fenduan.append(test_ini[item].split('\n\n'))

real_ini=[]
real_ini1={}
TASK_B_HEADER = [
    "PHYSICAL EXAMINATION",
    "PHYSICAL EXAM",
    "EXAM",
    "VITALS REVIEWED",
    "VITALS",
    "FAMILY HISTORY",
    "ALLERGIES",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "REVIEW OF SYSTEMS",
    "CURRENT MEDICATIONS",
    "PROCEDURE",
    "RESULTS",
    "MEDICATIONS",
    "INSTRUCTIONS",
    "IMPRESSION",
    "SURGICAL HISTORY",
    "SOCIAL HISTORY",
    "PLAN",
    "ASSESSMENT",
    "MEDICAL HISTORY",
]
for j,item in enumerate(fenduan):
    flag=0
    temp=''
    for i in range(len(item)):
        for item2 in TASK_B_HEADER:
            if item2 in item[i] and flag==1: 
                flag=2
                break
        if flag==2:
            break
        if flag==1:
            temp=temp+item[i]+'\n\n'
        if 'ASSESSMENT AND PLAN' in item[i]:
            flag=1
    real_ini.append(temp)
    real_ini1.setdefault(index_zhen[j],temp)


hpi_refine2=[]
index_zhen2=[]
ours2={}

with open(file_path_hpi,encoding='utf8') as f:
        i=0    
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            index_zhen2.append(row[0])
            hpi_refine2.append(row[1])
            ours2.setdefault(row[0],row[1])
lack2=[]
lack_c2=[]
for key,value in test_ini.items():
    if 'HISTORY OF PRESENT ILLNESS' in value and key in index_zhen2:
        print('TRUE')
    elif key in index_zhen:
        lack2.append(key)
        lack_c2.append(value)

fenduan2=[]
for item in index_zhen2:
    fenduan2.append(test_ini[item].split('\n\n'))
    
real_ini2=[]
real_ini22={}
TASK_B_HEADER = [
    "PHYSICAL EXAMINATION",
    "PHYSICAL EXAM",
    "EXAM",
    "ASSESSMENT AND PLAN",
    "VITALS REVIEWED",
    "VITALS",
    "FAMILY HISTORY",
    "ALLERGIES",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "REVIEW OF SYSTEMS",
    "CURRENT MEDICATIONS",
    "PROCEDURE",
    "RESULTS",
    "MEDICATIONS",
    "INSTRUCTIONS",
    "IMPRESSION",
    "SURGICAL HISTORY",
    "SOCIAL HISTORY",
    "PLAN",
    "ASSESSMENT",
    "MEDICAL HISTORY",
]

for j,item in enumerate(fenduan2):
    flag=0
    temp=''
    for i in range(len(item)):
        if j==42:
            print(item[2])
            temp=item[2].replace('HISTORY OF PRESENT ILLNESS\n','')
            break
        for item2 in TASK_B_HEADER:
            if item2 in item[i]: 
                flag=2
                break
        if flag==2:
            break
        if flag==1:
            temp=temp+item[i]+'\n\n'
        if 'HPI' in item[i]:
            flag=1
        if 'HISTORY OF PRESENT ILLNESS' in item[i]:
            flag=1
    real_ini2.append(temp)
    real_ini22.setdefault(index_zhen2[j],temp)
    
count=0
real_con={}
temp=[]
for key in test_ini:
    if key in index_zhen and key in index_zhen2:
        if real_ini1[key][:-2] in test_ini[key] and real_ini22[key][:-2] in test_ini[key]:
            real_con.setdefault(key,test_ini[key].replace(real_ini1[key][:-2],ours[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')).replace(real_ini22[key][:-2],ours2[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')))
        elif real_ini1[key][:-2] in test_ini[key] and real_ini22[key][:-2] not in test_ini[key]:
            real_con.setdefault(key,test_ini[key].replace(real_ini1[key][:-2],ours[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')))
        elif real_ini1[key][:-2] not in test_ini[key] and real_ini22[key][:-2] in test_ini[key]:
            real_con.setdefault(key,test_ini[key].replace(real_ini22[key][:-2],ours2[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')))
    elif key in index_zhen and key not in index_zhen2:
        if real_ini1[key][:-2] in test_ini[key]:
            real_con.setdefault(key,test_ini[key].replace(real_ini1[key][:-2],ours[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')))
    elif key not in index_zhen and key in index_zhen2:
        if real_ini22[key][:-2] in test_ini[key]:
            real_con.setdefault(key,test_ini[key].replace(real_ini22[key][:-2],ours2[key].replace('\n',' ').replace('   ',' ').replace('  ',' ')))
            
            
i=0
for key,value in test_ini.items():
    if key=='encounter_id':
        continue  
    elif key not in real_con.keys():
        real_con.setdefault(key,value.replace('\n',' ').replace('   ',' ').replace('  ',' '))
        
ref={}
ref_note=[]
with open('../data/ref.csv',encoding='utf8') as f:
        f_csv=csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            ref_note.append(row[1].replace('\n',' ').replace('   ',' ').replace('  ',' '))
            ref.setdefault(row[0],row[1].replace('\n',' ').replace('   ',' ').replace('  ',' '))
            
def save_csv(total,i,key,rname):
    with open(rname, 'a', newline='') as file:
        writer = csv.writer(file)
        if i==0:
            writer.writerow(["encounter_id", "note",])
        writer.writerow([key,total])
rname='formal_result.csv'

i=0
for key,value in ref.items():
    save_csv(real_con[key],i,key,rname)
    i=i+1