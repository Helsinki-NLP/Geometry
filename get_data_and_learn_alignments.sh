#!/bin/bash
:'
 USAGE: 
        source get_data /path/to/output/directory/
'
source /projappl/project_2001970/Geometry/env/bin/activate # MODIFY THIS PATH
currentdir=`pwd`
fastalign=/projappl/project_2001970/fast_align/build       # MODIFY THIS PATH
opustools=/projappl/project_2001970/OpusTools/opustools_pkg/bin              # MODIFY THIS PATH or delete this if you pip installed opus
#langlist=('bg' 'cs' 'da' 'de' 'el' 'es' 'et' 'fi' 'fr' 'hu' 'it' 'lt' 'lv' 'nl' 'pl' 'pt' 'ro' 'sk' 'sl' 'sv') 
langlist=('bg' 'cs' 'da' 'de' 'el' 'es' 'et' 'fi' 'fr' 'hu' 'it'           'nl'           'ro' 'sk'      'sv') # not all langs have MT models 
dbtypes=('test' 'train' 'dev')
outpath=${1:-"/scratch/project_2001970/Geometry/data/"}
cd $outpath
mkdir -p ${outpath}/alignments

for lang in ${langlist[@]}; do
   echo -e "\n####  downloading Europarl resources from opus for en-${lang}"

   $opustools/opus_express  -s en -t $lang --collections Europarl \
       --quality-aware --overlap-threshold 1.1 \
       --test-set  ${outpath}/en-${lang}.test \
       --dev-set   ${outpath}/en-${lang}.dev  \
       --train-set ${outpath}/en-${lang}.train \
       --test-quota 50000 \
       --dev-quota 2500  \
       --shuffle  
    
    echo "####  simple preprocessing "
    for db in ${dbtypes[@]}; do
        ${currentdir}/preprocess.sh en    < ${outpath}/en-${lang}.${db}.en       > ${outpath}/temp.en
        ${currentdir}/preprocess.sh $lang < ${outpath}/en-${lang}.${db}.${lang}  > ${outpath}/temp.tgt
        mv ${outpath}/temp.en  ${outpath}/en-${lang}.${db}.en 
        mv ${outpath}/temp.tgt ${outpath}/en-${lang}.${db}.${lang}

    done

    echo "####  getting datasets to 50000 utterances each for evaluation and exploratory analysis"
    mv ${outpath}/en-${lang}.test.en      ${outpath}/en-${lang}.en
    mv ${outpath}/en-${lang}.test.${lang} ${outpath}/en-${lang}.${lang}


  
    echo "...checking for empty lines in files $outpath/en-$lang.{en, $lang}"
    grep -v . $outpath/en-$lang.en | wc -l
    grep -v . $outpath/en-$lang.$lang | wc -l
    

    echo "####  full datasets - for supervised alignment method"
    # dev has alway 2500 and  en-$lang has 50000 (previously, the test)
    cat ${outpath}/en-${lang}.dev.en       ${outpath}/en-${lang}.en      ${outpath}/en-${lang}.train.en      > ${outpath}/en-${lang}.full.en
    cat ${outpath}/en-${lang}.dev.${lang}  ${outpath}/en-${lang}.${lang} ${outpath}/en-${lang}.train.${lang} > ${outpath}/en-${lang}.full.${lang}

    mkdir $outpath/en-${lang} # visual queue .... 
    :'
    echo "####  datasets for unsupervised alignment method"
    cat ${outpath}/en-${lang}.en      ${outpath}/en-${lang}.train.en      > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.en
    cat ${outpath}/en-${lang}.${lang} ${outpath}/en-${lang}.train.${lang} > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.${lang}


    #echo "####  directory sturcture needed for unsupervised method training"
    mkdir $outpath/en-${lang}
    ln -s ${outpath}/en-${lang}.train.en      ${outpath}/en-${lang}/train.source
    ln -s ${outpath}/en-${lang}.train.${lang} ${outpath}/en-${lang}/train.target
    ln -s ${outpath}/en-${lang}.dev.en        ${outpath}/en-${lang}/val.source
    ln -s ${outpath}/en-${lang}.dev.${lang}   ${outpath}/en-${lang}/val.target 
    
    mkdir $outpath/${lang}-en
    ln -s ${outpath}/en-${lang}.train.en      ${outpath}/${lang}-en/train.target
    ln -s ${outpath}/en-${lang}.train.${lang} ${outpath}/${lang}-en/train.source
    ln -s ${outpath}/en-${lang}.dev.en        ${outpath}/${lang}-en/val.target
    ln -s ${outpath}/en-${lang}.dev.${lang}   ${outpath}/${lang}-en/val.source 
    '
done
rm $outpath/*.ids



    echo "####  TRASH "

for lang in ${langlist[@]}; do
    head -n 2500 ${outpath}/en-${lang}.full.en > ${outpath}/en-${lang}.dev.en
    tail -n+2500 ${outpath}/en-${lang}.full.en | head -n 50000   > ${outpath}/en-${lang}.en
    tail -n+52500 ${outpath}/en-${lang}.full.en > ${outpath}/en-${lang}.train.en

    head -n 2500 ${outpath}/en-${lang}.full.$lang > ${outpath}/en-${lang}.dev.$lang
    tail -n+2500 ${outpath}/en-${lang}.full.$lang | head -n 50000   > ${outpath}/en-${lang}.$lang
    tail -n+52500 ${outpath}/en-${lang}.full.$lang > ${outpath}/en-${lang}.train.$lang
done
    cat ${outpath}/en-${lang}.dev.en       ${outpath}/en-${lang}.en      ${outpath}/en-${lang}.train.en      > ${outpath}/en-${lang}.full.en
    cat ${outpath}/en-${lang}.dev.${lang}  ${outpath}/en-${lang}.${lang} ${outpath}/en-${lang}.train.${lang} > ${outpath}/en-${lang}.full.${lang}

for lang in ${langlist[@]}; do
    ${currentdir}/preprocess.sh en    < ${outpath}/en-${lang}.dev.en       > ${outpath}/temp.en
    ${currentdir}/preprocess.sh $lang < ${outpath}/en-${lang}.dev.${lang}  > ${outpath}/temp.tgt
    mv ${outpath}/temp.en  ${outpath}/en-${lang}.dev.en 
    mv ${outpath}/temp.tgt ${outpath}/en-${lang}.dev.${lang}

    ${currentdir}/preprocess.sh en    < ${outpath}/en-${lang}.en       > ${outpath}/temp.en    
    ${currentdir}/preprocess.sh $lang < ${outpath}/en-${lang}.${lang}  > ${outpath}/temp.tgt
    mv ${outpath}/temp.en   ${outpath}/en-${lang}.en
    mv ${outpath}/temp.tgt  ${outpath}/en-${lang}.${lang}

    ${currentdir}/preprocess.sh en    <  ${outpath}/en-${lang}.train.en       > ${outpath}/temp.en
    ${currentdir}/preprocess.sh $lang <  ${outpath}/en-${lang}.train.${lang}  > ${outpath}/temp.tgt
    mv ${outpath}/temp.en   ${outpath}/en-${lang}.train.en
    mv ${outpath}/temp.tgt  ${outpath}/en-${lang}.train.${lang}
done

 echo "####  TRASH ENDS HERE ####"

module use -a /projappl/nlpl/software/modules/etc          # COMMENT LINE IF NOT IN PUHTI
module load nlpl-efmaral/0.1_20191218                      # COMMENT LINE IF NOT IN PUHTI - NEED TO SPECIFY WHERE IS align_eflomal.py


for lang in ${langlist[@]}; do
   echo "####  learning alignment for language pair: en-${lang}"

   srcfile=${outpath}/en-${lang}.full.en
   tgtfile=${outpath}/en-${lang}.full.${lang}

   align_eflomal.py  -s $srcfile -t $tgtfile \
                     -f ${outpath}/alignments/en-${lang}.align -r ${outpath}/alignments/en-${lang}.reverse.align 
     

   align_eflomal.py  -s $tgtfile -t $srcfile \
                     -f ${outpath}/alignments/${lang}-en.align -r ${outpath}/alignments/${lang}-en.reverse.align -v


done

# in PUHTI fastalign is needed by itself (there were environment issues with fastalign, when loading the eflomal module)
for lang in ${langlist[@]}; do
   ${fastalign}/atools -i ${outpath}/alignments/en-${lang}.align -j ${outpath}/alignments/en-${lang}.reverse.align -c intersect > ${outpath}/alignments/en-${lang}.intersect
   ${fastalign}/atools -i ${outpath}/alignments/${lang}-en.align -j ${outpath}/alignments/${lang}-en.reverse.align -c intersect > ${outpath}/alignments/${lang}-en.intersect
done



rm ${outpath}/alignments/??-en.train.reverse.align 
rm ${outpath}/alignments/??-en.dev.reverse.align 
rm ${outpath}/alignments/??-en.train.align 
rm ${outpath}/alignments/??-en.dev.align 

rm ${outpath}/alignments/en-??.train.reverse.align 
rm ${outpath}/alignments/en-??.dev.reverse.align 
rm ${outpath}/alignments/en-??.train.align 
rm ${outpath}/alignments/en-??.dev.align 