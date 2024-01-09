# REFLEXSE
a framework for clinical text generation
### Environment Description:
For Chinese:

    pip install -r chinese/requirements.txt

For english:

    pip install -r chinese/requirements.txt

### folders and files:


`/data` We provide preprocessed data, from aci-bench and IMCS-V2-MRG.

`/evaluation` Evaluation script, please use the command 'bert-score -r data/ref.txt  -c new_experiment/test_eval.txt --lang zh' for Chinese bertscore after running 'evaluation.py'. For details, please see https://github.com/tiiiger/bert_score

`/little_model` for training small_model.Please visit https://huggingface.co/patrickvonplaten/led-large-16384-pubmed to download the model to english/longformer_clinical. And visit 'https://drive.google.com/file/d/1XSXwz9MXO410-OWsIitc0hC2VmRMRg-K/view?usp=drive_link' to download the model 'ernie-health-chinese' to chinese/little_model/real_model/ernie-health-chinese. 

`/chinese/cc.py(hpi.py,suggestion.py)` is the script for chinese main experiment.


`/english/assessment.py(hpi.py)` is the script for chinese main experiment.



### Train

Please train the small model first, and then train the main experiment script.

For Chinese:

Please run chinese/little_model/real_model/ernie-health-chinese/little_model_cc.py(little_model_hpi.py or little_model_suggestion.py) to train the chinese rating model for different chapters.

For English:

Please run english/little_model/asse_model.py(hpi_model.py) to train the english rating model for different chapters.
