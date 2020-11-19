
'''
USAGE:
        source /projappl/project_2001970/Geometry/env/bin/activate
        scriptpath=/projappl/project_2001970/Geometry/code/utils
        srun --account=project_2001970 --partition=test --mem=32GB  --time=00:15:00  \
            python ${scriptpath}/re-save_MT_embs_by_layer.py 
'''
import h5py
import pickle
from tqdm import tqdm

tokdsentsname='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/embeddings/en-de_opus-mt-en-de_tokd.pkl' 
tokdsents=pickle.load(open(tokdsentsname,'rb'))
filename='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/embeddings/en-de_opus-mt-en-de.h5'

print('Loading h5 file')
h5f = h5py.File(filename, 'r')                                                                                                                            
setlen = len(tokdsents['source']) # .... len(h5f) is too slow
print('tokdsents', len(tokdsents['source']))

loaded_embs = [h5f.get(str(i))[()] for i in range(setlen)] 
#num_layers = loaded_embs[0].shape[0] #13

num_layers = 7
embs_by_layer = {f'layer{j}':[] for j in range(num_layers)}
print('Reshaping embeddings')
for sentidx in tqdm(range(setlen)):
    this_sent_embs = h5f.get(str(sentidx))[()] # [num_layers, sent_len, hdim]

    for layeridx in range(num_layers):
        embs_by_layer[f'layer{layeridx}'].append(this_sent_embs[layeridx])

h5f.close()

print('Saving...')
for key,value in tqdm(embs_by_layer.items()):
    outfile = f'/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/embs_by_layer/{key}_MTende.h5'

    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(value):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)



import sys
sys.exit(0)

#---------------

tokdsentsname='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embeddings/en-de_bert-base-cased_alignBERT2MTende_tokd.pkl' 
tokdsents=pickle.load(open(tokdsentsname,'rb'))
filename='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embeddings/en-de_bert-base-cased_alignBERT2MTende.h5'                                                                 
print('Loading h5 file')
h5f = h5py.File(filename, 'r')                                                                                                                            
setlen = len(tokdsents['source']) # .... len(h5f) is too slow

loaded_embs = [h5f.get(str(i))[()] for i in range(setlen)] 
#num_layers = loaded_embs[0].shape[0] #13

#num_layers = 13
LAYERS = [0, 1, 11, 12]
embs_by_layer = {f'layer{j}':[] for j in LAYERS}
print('Reshaping embeddings')
for sentidx in tqdm(range(setlen)):
    this_sent_embs = h5f.get(str(sentidx))[()] # [num_layers, sent_len, hdim]

    #for layeridx in range(num_layers):
    for layeridx in LAYERS:
        embs_by_layer[f'layer{layeridx}'].append(this_sent_embs[layeridx])

h5f.close()

print('Saving...')
for key,value in tqdm(embs_by_layer.items()):
    outfile = f'/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embs_by_layer/{key}_alignBERT2MTende.h5'

    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(value):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)



import sys
sys.exit(0)


tokdsentsname='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embeddings/en-de_opus-mt-en-de_alignMTende2BERT_tokd.pkl' 
tokdsents=pickle.load(open(tokdsentsname,'rb'))
filename='/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embeddings/en-de_opus-mt-en-de_alignMTende2BERT.h5'

print('Loading h5 file')
h5f = h5py.File(filename, 'r')                                                                                                                            
setlen = len(tokdsents['source']) # .... len(h5f) is too slow
print('tokdsents', len(tokdsents['source']))

loaded_embs = [h5f.get(str(i))[()] for i in range(setlen)] 
#num_layers = loaded_embs[0].shape[0] #13

num_layers = 7
embs_by_layer = {f'layer{j}':[] for j in range(num_layers)}
print('Reshaping embeddings')
for sentidx in tqdm(range(setlen)):
    this_sent_embs = h5f.get(str(sentidx))[()] # [num_layers, sent_len, hdim]

    for layeridx in range(num_layers):
        embs_by_layer[f'layer{layeridx}'].append(this_sent_embs[layeridx])

h5f.close()

print('Saving...')
for key,value in tqdm(embs_by_layer.items()):
    outfile = f'/scratch/project_2001970/Geometry/embeddings/Europarl_experiments/prealigned/embs_by_layer/{key}_alignMTende2BERT.h5'

    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(value):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)
