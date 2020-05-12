'''
Utilities for loading models - needed for embedding extraction
'''

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime 

import transformers
import torch

class bertModel():

    def __init__(self, bert_type='bert-base-uncased', cuda=False):
        self.N_BERT_LAYERS = 12
        self.ENC_DIM = 768

        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.model = BertModel.from_pretrained(bert_type)
        
        device='cuda' if cuda else 'cpu'
        self.device = device      
        self.model.eval()
        self.model.to(device)
    
    def tokenize(self, sentences):
        '''OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
                    # logging.basicConfig(level=logging.INFO) '''
        print(' | ',datetime.now().replace(microsecond=0), '|      tokenizing...')
        tokenized_sentences = []
        tokens_tensors = []
        for i in tqdm(range(len(sentences))):
            # add BERT tags
            sentences[i] = ' '.join(['[CLS]'] + sentences[i] + ['[SEP]'])
            tokenized_sentences.append(self.tokenizer.tokenize(sentences[i]))
            
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentences[i])
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensors.append(tokens_tensor)

        return tokens_tensors, tokenized_sentences


    def encode(self, tokens_tensors):
        print(' | ',datetime.now().replace(microsecond=0), '|      encoding...')
        if self.device =='cpu':
            print(' | ',datetime.now().replace(microsecond=0), '|      WARNING: using CPU... this might take a while.')
            print('                                        If you have a GPU capable device, use --cuda option.')
        encoded_sentences = []

        for i in tqdm(range(len(tokens_tensors))):
            # If you have a GPU, put everything on cuda
            tokens_tensors[i] = tokens_tensors[i].to(self.device)
            
            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensors[i])
            
            encoded_sentences.append(encoded_layers)
            
        return encoded_sentences

    def correct_bert_tokenization(self, bert_encodings, bert_sentences):
        
        print(' | ',datetime.now().replace(microsecond=0), '|      correcting for BERT subword tokenization...')
        corrected_sentences = []
        for bert_encoding, bert_sentence in tqdm(zip(bert_encodings, bert_sentences)):
            #print(bert_sentence)

            all_layers = []
            for layer in range(self.N_BERT_LAYERS):
                current_layer = []

                prev_token = bert_encoding[layer][0,0,:] # This is [CLS]!
                sequence_len = bert_encoding[layer].shape[1]

                accum = 1
                for token_id in range(1,sequence_len):
                    if  bert_sentence[token_id][:2] != '##':
                        current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling
                        accum = 1
                        prev_token = bert_encoding[layer][0,token_id,:]
                    else:
                        prev_token += bert_encoding[layer][0,token_id,:] # Average pooling
                        accum += 1
                # Add the final token too:
                current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling

                current_layer_tensor = torch.cat(current_layer, dim=1)
                # Get rid of the [CLS] and [SEP]
                current_layer_tensor = current_layer_tensor[:,1:-1,:]

                all_layers.append(current_layer_tensor)

            corrected_sentences.append(all_layers)
           
        #print('cbt: ', bert_sentences[0])
        #print('cbt: ', corrected_sentences[0][0].shape)

        return corrected_sentences


def load_onmt_model():
    pass

def load_huggingFace():
    import transformers 
    
