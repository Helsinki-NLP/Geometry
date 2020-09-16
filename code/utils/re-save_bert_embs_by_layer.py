
'''
USAGE:
        source /projappl/project_2001970/Geometry/env/bin/activate
        scriptpath=cd /projappl/project_2001970/Geometry/code/utils
        srun --account=project_2001970 --partition=test --mem=32GB  --time=00:15:00  \
            python ${scriptpath}/re-save_bert_embs_by_layer.py 
'''
import h5py
import pickle
from tqdm import tqdm


#tokdsentsname='/scratch/project_2001970/Geometry/embeddings/bert_13_layers/bert-base-cased_tokd.pkl' 
#tokdsents=pickle.load(open(tokdsentsname,'rb'))
filename='/scratch/project_2001970/Geometry/embeddings/bert_13_layers/bert-base-cased.h5'                                                                 
print('Loading h5 file')
h5f = h5py.File(filename, 'r')                                                                                                                            
setlen = 52082 #len(tokdsents) .... len(h5f) is too slow

loaded_embs = [h5f.get(str(i))[()] for i in range(setlen)] 
#num_layers = loaded_embs[0].shape[0] #13

num_layers = 13
embs_by_layer = {f'layer{j}':[] for j in range(num_layers)}
print('Reshaping embeddings')
for sentidx in tqdm(range(setlen)):
    this_sent_embs = h5f.get(str(sentidx))[()] # [num_layers, sent_len, hdim]

    for layeridx in range(num_layers):
        embs_by_layer[f'layer{layeridx}'].append(this_sent_embs[layeridx])

h5f.close()

print('Saving...')
for key,value in tqdm(embs_by_layer.items()):
    outfile = f'/scratch/project_2001970/Geometry/embeddings/bert_13_layers/embs_by_layer/{key}_bert-base-cased.h5'

    with h5py.File(outfile, 'w') as fout:
        for idx,embs in enumerate(value):
            fout.create_dataset(str(idx), embs.shape, dtype='float32', data=embs)