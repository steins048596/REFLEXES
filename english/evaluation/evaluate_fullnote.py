import sys
import json

import evaluate
import pandas as pd
import numpy as np

import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

def read_text( fn ) :
    texts = None
    if ".csv" in fn:
        temp_df=pd.read_csv(fn)
        print(temp_df["note"][0])
        print(type(temp_df["note"][0]))
        texts=[str(temp_df["note"][ind]).replace('\n',' ').replace('-',' ').replace('\u2022',' ').replace('    ',' ').replace('   ',' ').replace('  ',' ') for ind in range(len(temp_df))]
    else:
        with open( fn ) as f :
            texts = f.readlines()
    return texts



if len( sys.argv ) < 3 :
    print( 'usage: python evaluate_fullnote.py <gold> <sys> <metadata-file>' )
    sys.exit(0)


fn_gold = sys.argv[1]
fn_sys = sys.argv[2]
fn_metadata = [ sys.argv[3] if len( sys.argv )>3 else None ][0]


references = read_text( fn_gold )
predictions = read_text( fn_sys  )

predictions = [str(s) for s in predictions]
print( 'gold path: %s [%s summuaries]' %(fn_gold, len(references) ) )
print( 'system path: %s [%s summuaries]' %(fn_sys, len(predictions) ) )


if fn_metadata :
    df = pd.read_csv( fn_metadata )

    df[ 'reference' ] = references
    df[ 'prediction' ] = predictions

else :

    data = [ { 'id':ind, 'dataset':0, 'dialogue':'', 'reference':references[ind], 'prediction':predictions[ind]  } for ind in range( len( references ) ) ]
    df = pd.DataFrame( data )


num_test = len( df )



results_rouge_all = evaluate.load('rouge').compute(references=references, predictions=predictions, use_aggregator=False)


results_bertscore = evaluate.load('bertscore').compute(references=references, predictions=predictions, model_type='microsoft/deberta-xlarge-mnli')


results_all = { 
                "num_test":num_test,
                'ALL': { 'rouge1': np.mean( results_rouge_all['rouge1'][:num_test] ),
                          'rouge2': np.mean( results_rouge_all['rouge2'][:num_test] ),
                          'rougeL': np.mean( results_rouge_all['rougeL'][:num_test] ),
                          'rougeLsum': np.mean( results_rouge_all['rougeLsum'][:num_test] ),
                          'bertscore-precision': np.mean( results_bertscore['precision'][:num_test] ),
                          'bertscore-recall': np.mean( results_bertscore['recall'][:num_test] ),
                          'bertscore-f1': np.mean( results_bertscore['f1'][:num_test] )
                }
}

json_object = json.dumps( results_all, indent=4 )
fn_out = '{}.json'.format(fn_sys.split("/")[-1].split(".")[0])
with open( fn_out, 'w' ) as f :
    f.write( json_object )