#!/usr/bin/env python3 -u

"""
Compute similarity metrics for contextualized embeddings.
"""

from utils import opts
import ipdb

def  main(opts):
    pass


if __name__ == '__main__':
    parser = opts.get_parser()
    opt = parser.parse_args()
    if opt.debug_mode:
        with launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)
