
'''
in Joerg's: 
          source ~/.barshrc 
          source activate senteval
RUNNING WITH: 
python available_models/use_senteval.py \
          -model available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt \
          -src_lang en \
          -src fake \
          -tgt_lang de \
          -tgt fake \
          -gpu 0 

python available_models/use_senteval.py \
          -train_from available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt \
          -gpuid 0 \
          -use_attention_bridge True \
          -src_tgt FAKE \
          -data FAKE
'''

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import io
import numpy as np
import logging

import csv
# import onmt_utils

# Set PATHs
PATH_TO_SENTEVAL = '/scratch/project_2001970/SentEval'
#PATH_TO_INFERSENT='/home/local/vazquezj/git/InferSent'
PATH_TO_DATA = '/scratch/project_2001970/SentEval/data'


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import torch
import numpy as np

PATH_TO_ONMT='/scratch/project_2001970/AleModel/OpenNMT-py'
sys.path.insert(0, PATH_TO_ONMT)
PATH_TO_BPE="/projappl/project_2000945/subword-nmt/subword_nmt"
sys.path.insert(0, PATH_TO_BPE)
#sys.path.insert(0, PATH_TO_INFERSENT)
#import data

def invert_permutation(p):
    p = p.tolist()
    return torch.tensor([p.index(l) for l in range(len(p))])



#################################################
#   STEP1: import the trained model
##################################################

#import ipdb; ipdb.set_trace()
from onmt import inputters

import argparse
#parser = argparse.ArgumentParser(description='train.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#onmt.opts.add_md_help_argument(parser)
#onmt.opts.train_opts(parser)

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--only_save", action='store_true')
opt = parser.parse_args() 

if opt.cuda:
  torch.cuda.set_device(0)

checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
model_opt = checkpoint['opt']

fields = checkpoint['vocab']

if opt.cuda:
    torch.cuda.set_device(0)
    cur_device = "cuda"
    gpu_id=0
else:
    cur_device = "cpu"
    gpu_id=None
print("current device: ",cur_device)

from onmt import model_builder
model = model_builder.build_base_model(model_opt,fields,cur_device=='gpu',checkpoint,gpu_id)

model.to(cur_device)
model.eval()

from apply_bpe import BPE 
import codecs
codes = codecs.open("/scratch/project_2001970/AleModel/bpe-model.de-en-35k.wmt19-news-para.norm.tok.tc", encoding='utf-8')
bpe = BPE(codes)

def prepare(params, samples):
    #_, params.word2id = create_dictionary(samples)
    #params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    #params.wvec_dim = 300
    import ipdb; ipdb.set_trace()
    if params['save_embedds']:
        # get rid of empty lines
        samples = [sent if sent != [] else ['.'] for sent in samples]
        #  apply BPE to batch 
        sents=[]
        for sent in samples:
            str1 = ' '.join(sent)
            bpeversion=bpe.process_line(str1)
            sents.append(bpeversion)

        reader=inputters.str2reader['text'].from_opt(opt)    
        data = inputters.Dataset(fields=fields,
                                 data=[("src",sents)],
                                 readers=[reader],
                                 dirs=[None],
                                 sort_key=inputters.str2sortkey['text'])
        bsize=len(sents)#round(len(sents)/10)
        data_iter = inputters.OrderedIterator(
                dataset=data, device=cur_device,
                batch_size=bsize, train=False, sort=False,
                sort_within_batch=True, shuffle=False)

        #from onmt.translate.translator import Translator

        for BATCH in data_iter:
            permutation = BATCH.indices
            src, src_lengths = BATCH.src if isinstance(BATCH.src, tuple) \
                           else (BATCH.src, None)
            
            enc_states, memory_bank, src_lengths, layers = model.encoder(src, src_lengths)
        
        # reorder and convert to numpy
        for layer in layers:
            for key in layers[layer]:
                layers[layer][key] = layers[layer][key][invert_permutation(permutation),:,:].detach().numpy() #shape=[batch_size,att_heads,rnn_size]        
        
        #import ipdb; ipdb.set_trace(context=5)
        dumpable_dict={'sentences':sents}
        dumpable_dict.update(layers)
        import pickle
        with open('/projappl/project_2001970/'+params['current_task']+'_trf-6l-8ah.pt','wb') as f: 
            pickle.dump(dumpable_dict, f)

    return

def batcher(params, batch, key='EN'):
    
    if params['save_embedds']:
        import numpy as np
        return np.ones(len(batch))
    
    else:
        '''#----------------------------------------------------------
        # Onmt patch:
        #----------------------------------------------------------'''
        #    generate a temporal textfile to pass to Onmt modules

        batch = [sent if sent != [] else ['.'] for sent in batch]
        #  apply BPE to batch 
        sents=[]
        for sent in batch:
            str1 = ' '.join(sent)
            bpeversion=bpe.process_line(str1)
            sents.append(bpeversion)
        # batch is already lowercased, tokenized and normalized
        batch=sents
        

        reader=inputters.str2reader['text'].from_opt(opt)    
        data = inputters.Dataset(fields=fields,
                                 data=[("src",batch)],
                                 readers=[reader],
                                 dirs=[None],
                                 sort_key=inputters.str2sortkey['text'])

        #    generate iterator (of size 1) over the dataset
        bsize=len(batch)
        data_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=bsize, train=False, sort=False,
            sort_within_batch=True, shuffle=False)
        #    pass the batch information through the encoder
        import ipdb; ipdb.set_trace(context=5) 
        for BATCH in data_iter:
            #print('here_inside')
            permutation = BATCH.indices
            src = inputters.make_features(BATCH, side='src', data_type="text")
            src_lengths = None
            _, src_lengths = BATCH.src
            enc_states, memory_bank = model.encoders[index](src, src_lengths)
            enc_final, memory_bank = model.attention_bridge(memory_bank, src)
     
        import ipdb; ipdb.set_trace(context=5) 
        #print('out_again')
        memory_bank = memory_bank[:,invert_permutation(permutation),:] #shape=[att_heads,batch_size,rnn_size]
        #----------------------------------------------------------
        
        #import ipdb; ipdb.set_trace()
        output = memory_bank.transpose(0, 1).contiguous() #shape=[batch_size, att_heads,rnn_size]
        #output = output.view(output.size()[0], -1).detach()
        output = torch.mean(output, 1).detach()

        # make sure embeddings has 1 flattened M matrix per row.
        #memory_bank = memory_bank.transpose(0, 1).contiguous() #shape=[batch_size,att_heads,rnn_size]
        #embeddings = [mat.transpose(0,1).flatten().detach() for mat in memory_bank]
        #embeddings = np.vstack(embeddings)
        os.remove(batchfile)
        #return embeddings
        return output.cpu().numpy()

#####################
#   call SentEval
#####################

# Set params for SentEval
import random
#random.seed(1234)
sentevalseed = random.randint(1111, 9999)
print('using seed', sentevalseed)
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed': sentevalseed}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['save_embedds'] = opt.only_save
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    #params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
    #                                 'tenacity': 5, 'epoch_size': 4}   
    
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    transfer_tasks = ['STS16', 'STS14']

    results = se.eval(transfer_tasks)
    np.save(opt.save_model, results) 
    print(results)

# ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STSBenchmark', 'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

# source .bashrc
# source activate senteval
# cd git/OpenNMT-py-NI2_ALESSANDRO/
# python available_models/use_senteval.py -train_from available_models/someModels/en-cs.nil.model_step_15000.pt  -src_tgt en-cs     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-cs.nil.model_step_15000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/en-fr.nil.model_step_14000.pt  -src_tgt en-fr     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-fr.nil.model_step_14000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/en-de.nil.model_step_20000.pt    -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-de.nil.model_step_20000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt  -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_Many2EN-EN2Many.mono.model_step_82000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/multi-TO-multi.mono.model_step_91000.pt  -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_multi-TO-multi.mono.model_step_91000.OUT


