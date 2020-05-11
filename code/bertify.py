import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

def tokenize(sentence):
    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    sentence = ' '.join(['[CLS]'] + sentence + ['[SEP]'])
    print(sentence)
    tokenized_sentence = tokenizer.tokenize(sentence)

    print('tokenized: ', tokenized_sentence)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])

    return tokens_tensor


def encode(tokens_tensor):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    #tokens_tensor = tokens_tensor.to('cuda')
    #model.to('cuda')

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor)
        print(encoded_layers[0].shape)
