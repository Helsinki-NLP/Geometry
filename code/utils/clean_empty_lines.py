import sys
import os


if __name__=='__main__':
    src_in = open(sys.argv[1], 'r')
    tgt_in = open(sys.argv[2], 'r')

    src_out = open(sys.argv[1]+'.cln', 'w')
    tgt_out = open(sys.argv[2]+'.cln', 'w')

    src_lines = src_in.readlines()
    tgt_lines = tgt_in.readlines()

    for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        if src.strip() == '' or tgt.strip() == '':
            print('empty line at ', i+1)
            continue
        else:
            src_out.write(src)
            tgt_out.write(tgt)



