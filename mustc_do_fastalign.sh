#USAGE: source mustc_do_fastalign.sh

# load venv
source /projappl/project_2001970/Geometry/env/bin/activate

#path to fastalign
fastalign=/projappl/project_2001970/fast_align/build

# go to data dir
cd /projappl/project_2001970/Geometry/data

# data must be in format: src_sent ||| tgt_sent
paste mustc.train.ende50k.de mustc.train.ende50k.en  | sed 's/ *\t */ ||| /g' > mustc.train.ende50k.token

# run fast_align
${fastalign}/fast_align -i mustc.train.ende50k.token -d -o -v > mustc.train.ende50k.align
${fastalign}/fast_align -i mustc.train.ende50k.token -d -o -v -r > mustc.train.ende50k.reverse.align
${fastalign}/atools -i mustc.train.ende50k.align -j mustc.train.ende50k.reverse.align -c intersect > mustc.train.ende50k.intersect

