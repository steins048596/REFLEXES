# BeeRe

REFLEXSE(ENHANCING CLINICAL NOTE SUMMARIZATION: ITERATIVE REFLEXIONS WITH SMALL-MODEL SUPERVISION AND ERROR2CORRECT DEMONSTRATIONS)

### folders and files:



`/data` We provide preprocessed data, from aci-bench and IMCS-V2-MRG.

`/evaluation` Evaluation script, please use the command 'bert-score -r data/ref.txt  -c new_experiment/test_eval.txt --lang zh' for Chinese bertscore after running 'evaluation.py'. For details, please see https://github.com/tiiiger/bert_score

`/little_model` for training small_model

`/chinese/cc.py(hpi.py,suggestion.py)` is the script for chinese main experiment.


`/english/assessment.py(hpi.pyy)` is the script for chinese main experiment.



### Train

Please train the small model first, and then train the main experiment script.

## cite
@inproceedings{zhang-etal-2022-cblue,
    title = "{CBLUE}: A {C}hinese Biomedical Language Understanding Evaluation Benchmark",
    author = "Zhang, Ningyu and Chen, Mosha and Bi, Zhen and Liang, Xiaozhuan and Li, Lei and Shang, Xin and Yin, Kangping and Tan, Chuanqi and Xu, Jian and Huang, Fei and Si, Luo and Ni, Yuan and Xie, Guotong and Sui, Zhifang and Chang, Baobao and Zong, Hui and Yuan, Zheng and Li, Linfeng and Yan, Jun and Zan, Hongying and Zhang, Kunli and Tang, Buzhou and Chen, Qingcai",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = May,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.544",
    pages = "7888--7915"
}

@article{aci-bench,
  author = {Wen{-}wai Yim and
                Yujuan Fu and
                Asma {Ben Abacha} and
                Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}

@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}