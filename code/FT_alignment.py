import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from copy import deepcopy
from utils import model_loaders as Loader


use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using CUDA!")
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(non_blocking=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy


def keep_1to1(alignments):
    if len(alignments) == 0:
        return alignments
    
    counts1 = np.zeros(np.max(alignments[:,0]) + 1)
    counts2 = np.zeros(np.max(alignments[:,1]) + 1)
    
    for a in alignments:
        counts1[a[0]] += 1
        counts2[a[1]] += 1
    
    alignments2 = []
    for a in alignments:
        if counts1[a[0]] == 1 and counts2[a[1]] == 1:
            alignments2.append(a)
    return np.array(alignments2)

def load_alignments(align_path, max_sent, bad_idx):
    alignments = []
    with open(align_path) as align_file:
        """Lines should be of the form
        0-0 1-1 2-4 3-2 4-3 5-5 6-6
        
        Only keeps 1-to-1 alignments.
        
        Result:
        [
        [[0,0], [1,1], ...],
        ...
        ]
        """
        # need to only keep 1-1 alignments
        for i, line in enumerate(align_file):
            if i >= max_sent:
                break
            
            if i not in bad_idx:
                alignment = [pair.split('-') for pair in line.split()]
                alignment = np.array(alignment).astype(int)
                alignment = keep_1to1(alignment)
                alignments.append(alignment)

    return alignments

def load_align_corpus(sent_path, align_path, max_len = 64, max_sent = np.inf):
    sentences_1 = []
    sentences_2 = []
    bad_idx = []
    with open(sent_path) as sent_file:
        """Lines should be of the form
        doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
        
        Result: 
        sentences_1 = ['doch jetzt ist der Held gefallen .', ...  ]
        
        sentences_2 = ['but now the hero has fallen .', ... ]
        
        If sentences are already in sub-tokenized form, then max_len should be
        512. Otherwise, sentence length might increase after bert tokenization.
        (Bert has a max length of 512.)
        """
        for i, line in enumerate(sent_file):
            if i >= max_sent:
                break
            
            sent_1 = line[:line.index("|||")] #.split()
            sent_2 = line[line.index("|||"):].strip().strip('||| ') #.split()[1:]
            
            #if len(sent_1) > max_len or len(sent_2) > max_len:
            if len(sent_1.split()) > max_len or len(sent_2.split()) > max_len:
                bad_idx.append(i)
            else:
                sentences_1.append(sent_1)
                sentences_2.append(sent_2)
    
    if align_path is None:
        return [sentences_1, sentences_2, bad_idx]
    
    alignments = load_alignments(align_path, max_sent, bad_idx)
                
    return [sentences_1, sentences_2, alignments]
    
def partial_sums(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    arr.insert(0, 0)
    return arr[:-1]

def pick_aligned(sent_1, sent_2, align, cls_sep = True):
    """
    sent_1, sent_2 - lists of sentences. each sentence is a list of words.
    align - lists of alignments. each alignment is a list of pairs (i,j).
    """
    idx_1 = partial_sums([len(s) + 2 for s in sent_1]) #adjust indexes as if concat all sentences into a list of words
    idx_2 = partial_sums([len(s) + 2 for s in sent_2])
    align = [a + [i_1, i_2] for a, i_1, i_2 in zip(align, idx_1, idx_2)] # adjust alignment to new indexes
    align = reduce(lambda x, y: np.vstack((x, y)), align)
    align = align + 1 # to account for extra [CLS] at beginning
    
    if cls_sep:
        # also add cls and sep as alignments
        cls_idx = np.array(list(zip(idx_1, idx_2)))
        sep_idx = (cls_idx - 1)[1:]
        sep_idx_last = np.array([(sum([len(s) + 2 for s in sent_1]) - 1,
                        sum([len(s) + 2 for s in sent_2]) - 1)])
        align = np.vstack((align, cls_idx, sep_idx, sep_idx_last))
    
    # returns idx_1, idx_2
    # pick out aligned tokens using ann_1[idx_1], ann_2[idx_2]
    return align[:, 0], align[:, 1]

def align_bert_multiple(train, model, model_base, 
                        num_sentences, languages, batch_size, 
                        splitbatch_size = 4, epochs = 1,
                        learning_rate = 0.00005, learning_rate_warmup_frac = 0.1):
    # Adam hparams from Attention Is All You Need
    trainer = torch.optim.Adam([param for param in model.parameters() if
                                param.requires_grad], lr=1., 
                               betas=(0.9, 0.98), eps=1e-9)
                               
    # set up functions to do linear lr warmup
    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr
    learning_rate_warmup_steps = int(num_sentences * learning_rate_warmup_frac)
    warmup_coeff = learning_rate / learning_rate_warmup_steps
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= learning_rate_warmup_steps:
            print("Warming up, iter {}/{}".format(iteration, learning_rate_warmup_steps))
            set_lr(iteration * warmup_coeff)
            
    model_base.eval() # freeze and remember initial model # model_base should be the MT model for us
    
    total_processed = 0
    for epoch in range(epochs):
        for i in range(0, num_sentences, batch_size):
            loss = None
            model.train()
            schedule_lr(total_processed // (len(languages)))
            for j, language in enumerate(languages):
                sent_1, sent_2, align = train[j]
                ii = i % len(sent_1) # cyclic list - datasets may be diff sizes
                ss_1, ss_2 = sent_1[ii:ii+batch_size], sent_2[ii:ii+batch_size]
                aa = align[ii:ii+batch_size]
                
                # split batch to reduce memory usage
                for k in range(0, len(ss_1), splitbatch_size):
                    s_1 = ss_1[k:k+splitbatch_size]
                    s_2 = ss_2[k:k+splitbatch_size]
                    a = aa[k:k+splitbatch_size]
                    
                    # compute vectors for each position, pack the sentences
                    # result: packed_len x dim
                    b_tokd_1, ann_1 = model(deepcopy(s_1))
                    b_tokd_2, ann_2 = model(deepcopy(s_2))
                    hf_tokd, ann_2_base  = model_base(s_2)
                    # correct for subword tokenization
                    s_1_correct, ann_1_correct = model.correct_bert_tokenization(ann_1, b_tokd_1)
                    s_2_correct, ann_2_correct = model.correct_bert_tokenization(ann_2, b_tokd_2)
                    s_2_base_correct, ann_2_base_correct = model_base.correct_tokenization(hf_tokd, ann_2_base)

                    # pick out aligned indices in a packed representation
                    idx_1, idx_2 = pick_aligned(s_1_correct, s_2_correct, a)

                    loss_1 = F.mse_loss(ann_1[idx_1], ann_2_base[idx_2])
                    loss_2 = F.mse_loss(ann_2, ann_2_base)
                    loss_batch = loss_1 + loss_2
                    if loss is None: 
                        loss = loss_batch
                    else: 
                        loss += loss_batch
                total_processed += len(ss_1)
            
            print("Sentences {}-{}/{}, Loss: {}".format(
                    i, min(i+batch_size, num_sentences), num_sentences, loss))
            loss.backward()
            trainer.step()
            trainer.zero_grad()
                
    torch.save({'state_dict': model.state_dict(),
                'trainer' : trainer.state_dict(),}, 'best_network.pt')



def learn_alignments(data, languages, model, fastalign_path='/projappl/project_2001970/fast_align/build', moses_path='/projappl/project_2000945/mosesdecoder/scripts'):
    '''
    when entering here, data has not loaded the alignments
    '''
    import os
    align_paths = []
    tok_sent_paths = []
    for i, corpus in enumerate(data):
        sent_1, sent_2, _ = corpus
        _, s_1_tokd = model.tokenize(deepcopy(sent_1))
        _, s_2_tokd = model.tokenize(deepcopy(sent_2)) # this is in another language ... cannot tokenize with BERT ... 
        
        sentences = [' '.join(tokdsent1[1:-1]).replace(' ##','') + ' ||| ' + ' '.join(tokdsent2[1:-1]).replace(' ##','')  + ' \n' for tokdsent1, tokdsent2 in  zip(s_1_tokd,s_2_tokd) ]
        with open('./tempfile.{0}-en.txt'.format(languages[i]), 'w') as f:
            f.writelines(sentences)

        os.system('{0}/fast_align -i tempfile.{1}-en.txt -d -o -v > tempfile.{1}-en.align'.format(fastalign_path,languages[i]))
        os.system('{0}/fast_align -i tempfile.{1}-en.txt -d -o -v -r > tempfile.{1}-en.reverse.align'.format(fastalign_path,languages[i]))
        so.system('{0}/atools -i tempfile.{1}-en.align -j tempfile.{1}-en.reverse.align -c intersect > tempfile.{1}-en.intersect'.format(fastalign_path,languages[i]))

        align_paths.append('tempfile.{0}-en.intersect'.format(languages[i]))
        tok_sent_paths.append('tempfile.{0}-en.txt'.format(languages[i]))

    return tok_sent_paths, align_paths
        

# ENCODERS: 
#     BERT MODEL:  WITH TOKENIZER AND FORWARD PASS
#     MT MODEL:    WITH TOKENIZER AND FORWARD PASS

# DECODER:
#     MT MODEL: 

model = Loader.bertModel()
model_base = Loader.huggingfaceModel('Helsinki-NLP/opus-mt-en-de')

num_sent = 30 # size of experiment - to be splitted into train & dev
languages = ['de', 'bg']
sent_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.token.clean.reverse', 
              '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.token.clean.reverse']
align_paths = ['/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.de-en.intersect.reverse', 
               '/projappl/project_2001970/scripts/Cao_code/data/europarl-v7.bg-en.intersect.reverse']
batch_size = 8
num_epochs = 2

#data = [load_align_corpus(sent_path, align_path, max_sent = num_sent) for sent_path, align_path in zip(sent_paths, align_paths)]
#num_dev = 10
#dev = [(sent_1[:num_dev], sent_2[:num_dev], align[:num_dev]) for sent_1, sent_2, align in data]
#train = [(sent_1[num_dev:], sent_2[num_dev:], align[num_dev:]) for sent_1, sent_2, align in data]

# learn alignment
data_pre = [load_align_corpus(sent_path, None, max_sent = num_sent*100) for sent_path in sent_paths]
tok_sent_paths, align_paths = learn_alignments(data_pre, languages, model)

data = [load_align_corpus(sent_path, align_path, max_sent = num_sent) for sent_path, align_path in zip(tok_sent_paths, align_paths)]

#for i, corpus in enumerate(data):
#    bad_idx = corpus[2]
#    corpus[0], corpus[1] = corpus[0][:num_sent], corpus[1][:num_sent]
#    corpus[2] = load_alignments(align_paths[i], max_sent = num_sent, bad_idx = bad_idx)

num_dev = 10
dev = [(sent_1[:num_dev], sent_2[:num_dev], align[:num_dev]) for sent_1, sent_2, align in data]
train = [(sent_1[num_dev:], sent_2[num_dev:], align[num_dev:]) for sent_1, sent_2, align in data]


align_bert_multiple(train, model, model_base, num_sent, languages, batch_size, epochs=num_epochs)


