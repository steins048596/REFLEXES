# REFLEXSE
a framework for clinical text generation
### Environment Description:
Prepare environment for Chinese experiment,please use the command:

    pip install -r chinese/requirements.txt

Prepare environment for English experiment,please use the command:

    pip install -r english/requirements.txt

### folders and files:

`/data` We provide preprocessed data, from aci-bench and IMCS-V2-MRG.

`/evaluation` Evaluation script, please use the command 

    bert-score -r data/ref.txt  -c new_experiment/test_eval.txt --lang zh

for Chinese bertscore after running 'evaluation.py'. For details, please see https://github.com/tiiiger/bert_score

`/little_model` for training small_model.Please visit https://huggingface.co/patrickvonplaten/led-large-16384-pubmed to download the model to english/longformer_clinical. And visit 'https://drive.google.com/file/d/1XSXwz9MXO410-OWsIitc0hC2VmRMRg-K/view?usp=drive_link' to download the model 'ernie-health-chinese' to chinese/little_model/real_model/ernie-health-chinese. 



### Train small model

Please train the small model first, and then train the main experiment script.

Train Chinese small model:

    python chinese/little_model/real_model/ernie-health-chinese/little_model_cc.py
    python chinese/little_model/real_model/ernie-health-chinese/little_model_hpi.py 
    python chinese/little_model/real_model/ernie-health-chinese/little_model_suggestion.py

The model will be saved in chinese/little_model/real_model/few_shot_try

Train English small model:

    python english/little_model/asse_model.py
    python english/little_model/hpi_model.py

The model will be saved in english/little_model

### Start the runtime framework

For Chinese experiment:

    python chinese/cc.py
    python chinese/hpi.py
    python chinese/suggestion.py

For Chinese experiment:

    python english/assessment.py
    python english/hpi.py


