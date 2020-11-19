import gc
import time
import torch                                                            
import argparse                                                         
import transformers  

import numpy as np
from pathlib import Path                                                


from utils.FTalignment import (
    WordLevelBert,
    WordLevelOPUSmt, 
    align_bert_multiple, 
    load_align_corpus,
    evaluate_retrieval,
    )


def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    if args.dev_mode:
        update_args_to_devmode(args)

    num_sent = args.train_nsents
    num_dev = args.eval_nsents
    bsz = args.batch_size
    num_epochs = args.epochs
    data_path = args.data_path
    data_path_tgt = args.data_path_tgt
    align_path = args.alignment_file


    if not args.only_run_tests:
        if align_path:
            print('Loading parallel corpus', flush=True)
            data = load_align_corpus(data_path, data_path_tgt, align_path, max_sent = num_sent+num_dev)
        else:
            data = load_4_aligning(f'{args.data_path}', max_sent = num_sent+num_dev)
       
        dev   = [(data[0][:num_dev], data[1][:num_dev], data[2][:num_dev])]
        train = [(data[0][num_dev:], data[1][num_dev:], data[2][num_dev:])]


        alignedBERT_state_dict = do_finetuning_alignment(
                    train,
                    dev,
                    model=args.model,
                    model_base=args.model_base,
                    num_sent=args.train_nsents,
                    bsz=args.batch_size,
                    num_epochs=args.epochs,
                    align_to_model_decoder=args.align_to_model_decoder,
                    align_to_model_base_decoder=args.align_to_model_base_decoder,
                    outdir=args.outdir ,
                    log_every_n_batches=args.log_every_n_align
                )
    else:
        if align_path:
            print('Loading parallel corpus', flush=True)
            data = load_align_corpus(data_path, data_path_tgt, align_path, max_sent = num_dev)
        else:
            data = load_4_aligning(f'{data_path}', max_sent = num_dev)
       
        dev   = [(data[0][:num_dev], data[1][:num_dev], data[2][:num_dev])]
        #train = [(data[0][num_dev:], data[1][num_dev:], data[2][num_dev:])]

        model_base = get_model(args.model_base, args.align_to_model_base_decoder)
        model = get_model(args.model, args.align_to_model_decoder, model_base.dim)

        print("Word retrieval accuracy:", evaluate_retrieval(dev[0], model, model_base))
        model_state_dict = torch.load(args.load_aligned_model, map_location=torch.device(model.device))
        model.load_state_dict(model_state_dict['state_dict'])
        print("Word retrieval accuracy after alignment:", evaluate_retrieval(dev[0], model, model_base))
        
        
        
def load_4_aligning(
    sent_path,
    max_len = 64, 
    max_sent = 250000,
    ):
    '''
    This function emulates load_align_corpus from utils.FTalignment.py
    To use when wanting to encode english sentences with different models, 
    instead of encoding different language sentences with the same model
    '''
    import nltk
    sentences_1 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        for i, line in enumerate(sent_file):
            if i >= max_sent:
                break
          
            sent_1 = nltk.TweetTokenizer().tokenize(line)    
            if len(sent_1) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)

        alignments = [   np.array([  [j,j] for j,_ in enumerate(sent)  ])   for sent  in sentences_1   ]
        return sentences_1, sentences_1, alignments

def get_model(
    modelname:str, 
    use_decoder:bool=False, 
    project_to_dim:int=512
    ):
    if modelname.find('opus-mt') > -1:
        if not use_decoder:    
            model = WordLevelOPUSmt(modelname)
        else: 
            raise NotImplementedError
            #model = WordLevelOPUSmt_decoder(modelname)
    else:
        model = WordLevelBert(modelname, modelname=='bert-base-uncased', project_to_dim)
    
    return model

def do_finetuning_alignment(
    train,
    dev,
    model:str,
    model_base:str,
    num_sent:int=250000, 
    bsz:int=16, 
    num_epochs:int=10,
    align_path:str=None,
    align_to_model_decoder:bool=False,
    align_to_model_base_decoder:bool=False,
    outdir:str='./',
    log_every_n_batches:int=100,
     ):
    '''
    Align embeddings of model with the embeddings of model_base using Cao's alignment method
    IN:
        - model      [str]: name of huggingface model to use 
        - model_base [str]: name of huggingface model to use as anchor for alignment
        - data_path  [str]: path to src sentences.
        - data_path_tgt  [str]: path to tgt sentences.
                             Lines should be of the form     doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
        - align_path [str]: path to alignement file. Only keeps 1-to-1 alignments. 
                            Lines should be of the form      0-0 1-1 2-4 3-2 4-3 5-5 6-6

    OUT:
        - aligned_state_dict [OrderedDict]: statedict of a BERT model with new parameters
    '''
    print(f'Running alignment routine:')
    print(f'        embeddings from model \"{model}\" are to be aligned with the ones from model \"{model_base}\" ')
    print(f'        training with {num_sent} sentences, for {num_epochs} epochs ')
    print(f'        checkpoints will be saved to  {outdir}/alignment_network_ckpt_XX.pt \n', flush=True)

    print(f'Loading models:')
    model_base = get_model(model_base, align_to_model_base_decoder) # this has been the MT encoder for us
    model = get_model(model, align_to_model_decoder, model_base.dim) # this is the one that will have the parameters updated
    

    
    #import ipdb; ipdb.set_trace()
    #print("Word retrieval accuracy:", evaluate_retrieval(dev[0], model, model_base))

    align_bert_multiple(
        train, 
        model, 
        model_base, 
        num_sentences=num_sent, 
        languages=['dummy'], 
        batch_size=bsz, 
        epochs=num_epochs, 
        outdir=outdir,
        log_every_n_batches=log_every_n_batches,
        devdata = dev[0],
    )
    
    print("Word retrieval accuracy after alignment:", evaluate_retrieval(dev[0], model, model_base))


    #free memory
    aligned_state_dict = model.state_dict()
    del model, model_base
    gc.collect()
    return aligned_state_dict 



def update_args_to_devmode(args):
    print('dev mode activated... overriding parameters:', flush=True)
    print('                           --train_nsents = 124,')
    print('                           --eval_nsents = 24,')
    print('                           --batch_size = 2,')
    print('                           --epochs = 1')
    print('                           --log_every_n_align = 1')

    args.train_nsents = 124
    args.eval_nsents = 24
    args.batch_size = 2
    args.epochs = 1
    args.log_every_n_align = 1

def add_argpars_args(parser):
    # PATHS
    parser.add_argument("--outdir", type=str, default='../outputs/', help="Path to directory where aligned models will be stored")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
    parser.add_argument("--data_path_tgt", type=str, required=False,  default=None, 
                         help="Path to target data file\
                               Use when the alignment is to be done accross different languages. ")

    # TRAINING PARAMETERS
    parser.add_argument("--train_nsents", type=int, default=250000, help="Number of sentences used to train Cao's alignment method.")
    parser.add_argument("--eval_nsents", type=int, default=1024, help="Number of sentences used as val/eval.")
    parser.add_argument("--test_nsents", type=int, default=1024, help="Number of sentences used as test.")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for learning Cao's alignment method.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size used for train, dev and test.")

    # MODELS
    parser.add_argument("--model_base", type=str, required=True,
                        help="Model to be use as anchor for alignment \
                              Can choose between huggingface bert-base-cased, bert-base-uncased and any of Helsinki-NLP/opus-mt-$src-$tgt")
    parser.add_argument("--align_to_model_base_decoder", action="store_true", 
                        help="When --model_base is an MT model: Helsinki-NLP/opus-mt-$src-$tgt, use this flag to align to the decoder embeddings\
                              instead of aligning to the encoder embeddings.\
                        ATTENTION: Not yet Implemented")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to be finetuned to be aligned to the model_base enmbeddings\
                              Can choose between huggingface bert-base-cased, bert-base-uncased and any of Helsinki-NLP/opus-mt-$src-$tgt")
    parser.add_argument("--align_to_model_decoder", action="store_true", 
                        help="When --model is an MT model: Helsinki-NLP/opus-mt-$src-$tgt, use this flag to align the decoder embeddings \
                              towards --model_base instead of aligning the encoder embeddings.\
                        ATTENTION: Not yet Implemented")
    parser.add_argument("--load_aligned_model", type=str, required=False,  default=None,
                         help="Path to a prealigned model\
                               will be loaded to test in wrod retrieval task")
    
    
    # ALIGNMENT
    parser.add_argument("--alignment_file", type=str, required=False, 
                        help="Path to word alignment file. fast-align format: one sentence per line. \
                              Each line contains src-tgt word indices corresponding to the train set, e.g. 0-0 1-2 3-5 4-4 ...\
                        ATTENTION: Needed when the alignment is to be done accross different languages.\
                                When not specified, model and model_base encoddings are assumed to be the same language \
                                and the aligment is emulated by generating the identity, i.e. 0-0 1-1 2-2 3-3- 4-4 ...")

    # OTHER
    parser.add_argument("--dev_mode", action="store_true", 
                        help="Run in development mode: parameters to be overwritten. \
                        ATTENTION: ipdb to be launched on exception ")

    parser.add_argument("--log_every_n_align", type=int, default=250, 
                        help="How often to report results when doing Cao's alignment method.") 

    parser.add_argument("--only_run_tests", action="store_true", 
                        help="Run word retrieval with the hugginface models and a given model\
                              given with the flag --load_aligned_model  ")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to train alignement between bert models and MT encoder/decoder embeddings')
    
    parser = add_argpars_args(parser) 
    args = parser.parse_args()  
    
    if args.dev_mode:
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            main(args)
    else:
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            main(args)


'''
source  /projappl/project_2001970/Geometry/env/bin/activate

cd /projappl/project_2001970/Geometry/code

## ALIGN BERT embeddings to MT encoder embeddings (both in english)
#run --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
python do_FT_alignment.py --data_path   /scratch/project_2001970/Geometry/data/en-de.full.en      \
                          --model        bert-base-cased                                          \
                          --model_base   Helsinki-NLP/opus-mt-en-de                               \
                          --train_nsents 200000 \
                          --epochs 1 \
                          --outdir      /scratch/project_2001970/Geometry/aligned_models/intento/align_BERT_2_MTende #--dev_mode


## ALIGN MT encoder embeddings to BERT embeddings  (both in english)
#srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
python do_FT_alignment.py --data_path   /scratch/project_2001970/Geometry/data/en-de.full.en      \
                          --model        Helsinki-NLP/opus-mt-en-de                               \
                          --model_base   bert-base-cased                                          \
                          --train_nsents 200000 \
                          --epochs 1 \
                          --outdir      /scratch/project_2001970/Geometry/aligned_models/intento/align_MTende_2_BERT #--dev_mode



## ALIGN BERT embeddings (english) to german MT encoder embeddings 
#srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
python do_FT_alignment.py --data_path      /scratch/project_2001970/Geometry/data/en-de.full.en       \
                          --data_path_tgt  /scratch/project_2001970/Geometry/data/en-de.full.de       \
                          --model           bert-base-cased                                           \
                          --model_base      Helsinki-NLP/opus-mt-de-en                                \
                          --alignment_file /scratch/project_2001970/Geometry/data/alignments/en-de.intersect \
                          --train_nsents 200000 \
                          --epochs 1 \
                          --outdir         /scratch/project_2001970/Geometry/aligned_models/intento/align_BERT_2_MTdeen #\
                          #--dev_mode
                          

## ALIGN german MT encoder embeddings to english BERT embeddings
#srun --account=project_2001970 --partition=gputest --gres=gpu:v100:1,nvme:2 --mem=8GB --time=00:15:00 \
srun --account=project_2001970 --partition=gpu --gres=gpu:v100:1,nvme:2 --mem=64GB --time=02:00:00 \
python do_FT_alignment.py --data_path      /scratch/project_2001970/Geometry/data/en-de.full.de       \
                          --data_path_tgt  /scratch/project_2001970/Geometry/data/en-de.full.en       \
                          --model           Helsinki-NLP/opus-mt-de-en                                \
                          --model_base      bert-base-cased                                           \
                          --alignment_file /scratch/project_2001970/Geometry/data/alignments/de-en.intersect \
                          --train_nsents 200000 \
                          --epochs 1 \
                          --outdir         /scratch/project_2001970/Geometry/aligned_models/intento/align_MTdeen_2_BERT  #\
                          #--dev_mode



'''