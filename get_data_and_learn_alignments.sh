#!/bin/bash
:'
 USAGE: 
        source get_data /path/to/output/directory/
'
source /projappl/project_2001970/Geometry/env/bin/activate # MODIFY THIS PATH
fastalign=/projappl/project_2001970/fast_align/build       # MODIFY THIS PATH
opustools=/projappl/project_2001970/OpusTools/opustools_pkg/bin              # MODIFY THIS PATH or delete this if you pip installed opus
# langlist=('bg' 'cs' 'da' 'de' 'el' 'es' 'et' 'fi' 'fr' 'hu' 'it' 'lt' 'lv' 'nl' 'pl' 'pt' 'ro' 'sk' 'sl' 'sv') 
langlist=('bg' 'cs' 'da' 'de'      'es' 'et' 'fi' 'fr' 'hu' 'it'                          'ro'           'sv'  'el' 'nl' 'sk' ) # not all langs have MT models 

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

    echo "####  getting datasets to 50000 utterances each for training supervised alignment method"
    mv ${outpath}/en-${lang}.test.en      ${outpath}/en-${lang}.en
    mv ${outpath}/en-${lang}.test.${lang} ${outpath}/en-${lang}.${lang}
  
    echo "...checking for empty lines in files $outpath/en-$lang.{en, $lang}"
    grep -v . $outpath/en-$lang.en | wc -l
    grep -v . $outpath/en-$lang.$lang | wc -l
    
    echo "####  train datasets for unsupervised alignment method"
    cat ${outpath}/en-${lang}.en      ${outpath}/en-${lang}.train.en      > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.en
    cat ${outpath}/en-${lang}.${lang} ${outpath}/en-${lang}.train.${lang} > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.${lang}

    rm ${outpath}/en-${lang}.test.?? 

    mkdir $outpath/en-${lang}
    ln -s ${outpath}/en-${lang}.train.en      ${outpath}/en-${lang}/train.source
    ln -s ${outpath}/en-${lang}.train.${lang} ${outpath}/en-${lang}/train.target
  
    ln -s ${outpath}/en-${lang}.dev.en      ${outpath}/en-${lang}/val.source
    ln -s ${outpath}/en-${lang}.dev.${lang} ${outpath}/en-${lang}/val.target 

done
rm $outpath/*.ids

:' NOT NEEDED IN GIT VERSION:

for lang in ${langlist[@]}; do
   echo -e "\n####   for en-${lang}"


    head -n 2500 ${outpath}/en-${lang}.test.en      > ${outpath}/en-${lang}.val.en
    head -n 2500 ${outpath}/en-${lang}.test.${lang} > ${outpath}/en-${lang}.val.${lang}
 
    tail -n 22500 ${outpath}/en-${lang}.test.en      > ${outpath}/en-${lang}.tmp.en
    tail -n 22500 ${outpath}/en-${lang}.test.${lang} > ${outpath}/en-${lang}.tmp.${lang}

    cat ${outpath}/en-${lang}.tmp.en      ${outpath}/en-${lang}.dev.en      ${outpath}/en-${lang}.train.en      > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.en
    cat ${outpath}/en-${lang}.tmp.${lang} ${outpath}/en-${lang}.dev.${lang} ${outpath}/en-${lang}.train.${lang} > ${outpath}/temp.txt
    mv ${outpath}/temp.txt ${outpath}/en-${lang}.train.${lang}
    
    mv ${outpath}/en-${lang}.val.en      ${outpath}/en-${lang}.dev.en
    mv ${outpath}/en-${lang}.val.${lang} ${outpath}/en-${lang}.dev.${lang}


    rm ${outpath}/en-${lang}.tmp.?? ${outpath}/en-${lang}.test.??

  mkdir $outpath/en-${lang}
  ln -s ${outpath}/en-${lang}.train.en      ${outpath}/en-${lang}/train.source
  ln -s ${outpath}/en-${lang}.train.${lang} ${outpath}/en-${lang}/train.target
  
  ln -s ${outpath}/en-${lang}.dev.en      ${outpath}/en-${lang}/val.source
  ln -s ${outpath}/en-${lang}.dev.${lang} ${outpath}/en-${lang}/val.target

done


for lang in ${langlist[@]}; do
    
    head -n 25000 ${outpath}/en-${lang}.en > ${outpath}/en-${lang}.test.en 
    tail -n 25000 ${outpath}/en-${lang}.en > ${outpath}/en-${lang}.dev.en 

    head -n 25000 ${outpath}/en-${lang}.${lang}  > ${outpath}/en-${lang}.test.${lang}  
    tail -n 25000 ${outpath}/en-${lang}.${lang}  > ${outpath}/en-${lang}.dev.${lang} 

done
'

module use -a /projappl/nlpl/software/modules/etc          # COMMENT LINE IF NOT IN PUHTI
module load nlpl-efmaral/0.1_20191218                      # COMMENT LINE IF NOT IN PUHTI - NEED TO SPECIFY WHERE IS align_eflomal.py
for lang in ${langlist[@]}; do
   echo "####  learning alignment for language pair: en-${lang}"
   align_eflomal.py  -s ${outpath}/en-${lang}.train.en -t ${outpath}/en-${lang}.train.${lang} \
                     -f ${outpath}/alignments/en-${lang}.train.en.align -r ${outpath}/alignments/en-${lang}.train.en.reverse.align -v
   ${fastalign}/atools -i ${outpath}/alignments/en-${lang}.train.en.align -j ${outpath}/alignments/en-${lang}.train.en.reverse.align -c intersect > ${outpath}/alignments/en-${lang}.train.en.intersect

done
#for lang in ${langlist[@]}; do
   
    #opus_read --directory Europarl \
    #--source en --target ${lang} \
    #--src_range 1 --tgt_range 1 -m 56000 \
    #--attribute certainty --threshold 1.1 \
    #--write ${outpath}/en-${lang}.txt
   
   #awk "/^\(src\)/" ${outpath}/en-${lang}.txt | sed "s/^.*\">//" > $outpath/en-${lang}.en.txt
   #awk "/^\(trg\)/" ${outpath}/en-${lang}.txt | sed "s/^.*\">//" > $outpath/en-${lang}.en.txt

#done