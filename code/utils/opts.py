
"""
Simple argument parser. Can specify model to load, metrics to compute, datasets to use, sampling size, treatment of of tokenization & subword units, etc.
"""

import argparse

def get_parser():
    """
    Add arguments in alphabetical order.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--apply_bpe", action='store_true',
                        help="Does your model needs BPEd data as input?")

    parser.add_argument("--bpe_model", required=False,
                        help='path to read onmt_model')

    parser.add_argument("--cuda", action='store_true',
                        help="Whether to use a cuda device.")
    
    parser.add_argument("--data_path", required=False, type=str, default="../data/allSTS.txt",
                        help="Dataset used to sample from. Defaults to STS datasets.")

    parser.add_argument("--debug_mode", action='store_true',
                        help="Launch ipdb debugger if script crashes.")
   
 
    parser.add_argument("--intrasentsim_samplesize", "-intra_ss", required=False, type=int, default=500,
                        help='size of the sentence sample to be extracted for computing the intra-sentence similarity')

    parser.add_argument("--model", required=False,
                        help='path to read onmt_model')


    parser.add_argument("--selfsim_samplesize", "-self_ss",required=False, type=int, default=2500,
                        help='size of the words sample to extract for computing the self-similarity')

    parser.add_argument("--sample_size", required=False, type=int, default=1000,
                        help='size of the words sample to be extracted for computing the isotropy correction factor')

    return parser
