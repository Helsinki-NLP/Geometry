#!/bin/bash

head /scratch/project_2001970/Geometry/data/align.news.en-de.de.txt
head /scratch/project_2001970/Geometry/data/align.news.en-de.en.txt

head /scratch/project_2001970/Geometry/en_de_europarl
head /scratch/project_2001970/Geometry/en_de_europarl


head -n 500 /scratch/project_2001970/Geometry/data/align.news.en-de.de.txt > /scratch/project_2001970/Geometry/data/align.news.en-de.dev.de.txt
head -n 500 /scratch/project_2001970/Geometry/data/align.news.en-de.en.txt > /scratch/project_2001970/Geometry/data/align.news.en-de.dev.en.txt
tail -n+501 /scratch/project_2001970/Geometry/data/align.news.en-de.de.txt > /scratch/project_2001970/Geometry/data/align.news.en-de.train.de.txt
tail -n+501 /scratch/project_2001970/Geometry/data/align.news.en-de.en.txt > /scratch/project_2001970/Geometry/data/align.news.en-de.train.en.txt

head -n 1500 /scratch/project_2001970/Geometry/en_de_europarl/en-de.dev.de.cln > /scratch/project_2001970/Geometry/en_de_europarl/en-de.shortdev.de.cln
head -n 1500 /scratch/project_2001970/Geometry/en_de_europarl/en-de.dev.en.cln > /scratch/project_2001970/Geometry/en_de_europarl/en-de.shortdev.en.cln

outdir=/scratch/project_2001970/Geometry/en_de_data4finalmodel
# validation: 1500 from europarl, 1423 from must-c and 500 from newstest = 3423
cat /scratch/project_2001970/Geometry/data/align.news.en-de.dev.en.txt \
    /scratch/project_2001970/Geometry/en_de_europarl/en-de.shortdev.en.cln \
    /scratch/project_2001970/Geometry/en_de/val.source > ${outdir}/val.source

cat /scratch/project_2001970/Geometry/data/align.news.en-de.dev.de.txt \
    /scratch/project_2001970/Geometry/en_de_europarl/en-de.shortdev.de.cln \
    /scratch/project_2001970/Geometry/en_de/val.target > ${outdir}/val.target


# train unsupervised alignment: 150K from must-c, 13k from newstests and 150k from europarl
head -n 150000 /scratch/project_2001970/Geometry/en_de/train.source > $outdir/mustc.train.source.temp
head -n 150000 /scratch/project_2001970/Geometry/en_de/train.target > $outdir/mustc.train.target.temp

head -n 150000 /scratch/project_2001970/Geometry/en_de_europarl/en-de.train.en.cln > $outdir/europarl.train.source.temp
head -n 150000 /scratch/project_2001970/Geometry/en_de_europarl/en-de.train.de.cln > $outdir/europarl.train.target.temp

cat /scratch/project_2001970/Geometry/data/align.news.en-de.train.en.txt \
    $outdir/europarl.train.source.temp \
    $outdir/mustc.train.source.temp > ${outdir}/train.source


cat /scratch/project_2001970/Geometry/data/align.news.en-de.train.en.txt \
    $outdir/europarl.train.target.temp \
    $outdir/mustc.train.target.temp > ${outdir}/train.target

grep -v . $outdir/train.target | wc -l
grep -v . $outdir/train.source | wc -l
grep -v . $outdir/val.target | wc -l
grep -v . $outdir/val.source | wc -l

# train supervised alignment: 45k from must-c, 13k from newstest and 45k from europarl: 103k
head -n 45000 /scratch/project_2001970/Geometry/en_de/train.source > $outdir/mustc.train.source.temp
head -n 45000 /scratch/project_2001970/Geometry/en_de/train.target > $outdir/mustc.train.target.temp

head -n 45000 /scratch/project_2001970/Geometry/en_de_europarl/en-de.train.en.cln > $outdir/europarl.train.source.temp
head -n 45000 /scratch/project_2001970/Geometry/en_de_europarl/en-de.train.de.cln > $outdir/europarl.train.target.temp

cat /scratch/project_2001970/Geometry/data/align.news.en-de.train.en.txt \
    $outdir/europarl.train.source.temp \
    $outdir/mustc.train.source.temp > ${outdir}/trainalignment.source


cat /scratch/project_2001970/Geometry/data/align.news.en-de.train.en.txt \
    $outdir/europarl.train.target.temp \
    $outdir/mustc.train.target.temp > ${outdir}/trainalignment.target

grep -v . $outdir/trainalignment.target | wc -l
grep -v . $outdir/trainalignment.source | wc -l

shuffle_files () {
      srcin=$1
      tgtin=$2

python /projappl/project_2001970/Geometry/code/utils/shuffle_files.py $srcin $tgtin

}

shuffle_files ${outdir}/trainalignment.source ${outdir}/trainalignment.target
shuffle_files ${outdir}/train.source          ${outdir}/train.target

cat ${outdir}/val.source   ${outdir}/trainalignment.source.shf > ${outdir}/trainalignment.source
cat ${outdir}/val.target  ${outdir}/trainalignment.target.shf > ${outdir}/trainalignment.target
#mv ${outdir}/trainalignment.source.shf  ${outdir}/trainalignment.source
#mv ${outdir}/trainalignment.target.shf  ${outdir}/trainalignment.target
mv ${outdir}/train.source.shf        ${outdir}/train.source          
mv ${outdir}/train.target.shf        ${outdir}/train.target

#cp   ${outdir}/trainalignment.source ${outdir}/trainalignment.source.shf
#cp   ${outdir}/trainalignment.target ${outdir}/trainalignment.target.shf