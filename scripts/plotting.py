import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

#INFILE = '../results/self_similarities_1000.pkl'
INFILE = '../results/self_similarities_1000_cased.pkl'
SAVEFILE = '../results/visualization/bert_selfsim.png'

norm = pickle.load(open("../results/normalized1000.p",'rb'))
norm = np.transpose(norm)
avg_norm = np.mean(norm, 1)

baselines = np.array([[0.28299081],
       [0.31967735],
       [0.32831138],
       [0.35364175],
       [0.37922698],
       [0.37682441],
       [0.37625521],
       [0.37906641],
       [0.36455202],
       [0.37561709],
       [0.4218004 ],
       [0.37211022]])

print(baselines.shape)

self_similarities = pickle.load(open(INFILE,'br'))
avg_similarities = np.mean(self_similarities, 1).reshape((12,1)) 

print(avg_similarities.shape)

avg_similarities = 2 * avg_similarities - baselines
print(avg_similarities.shape)

layers = range(1,12+1)
fig = plt.plot(layers, avg_similarities, '-.')

#plt.plot(range(0, 7), 2*norm, '-.')


plt.title('Anisotrophy-Corrected Avg Cosine Similarity (1000 Random Words)')
plt.legend(['BERT', 'MT embeddings'])
plt.ylim([0, 1])
plt.xticks(layers)
plt.grid(True)
plt.savefig(SAVEFILE)
