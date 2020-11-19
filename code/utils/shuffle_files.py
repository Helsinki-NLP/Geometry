import sys
import os
import sklearn

if __name__=='__main__':
    src_in = open(sys.argv[1], 'r')
    tgt_in = open(sys.argv[2], 'r')

    src_out = open(sys.argv[1]+'.shf', 'w')
    tgt_out = open(sys.argv[2]+'.shf', 'w')

    src_lines = src_in.readlines()
    tgt_lines = tgt_in.readlines()

    src_lines, tgt_lines = sklearn.utils.shuffle(src_lines, tgt_lines)

    for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        src_out.write(src)
        tgt_out.write(tgt)



