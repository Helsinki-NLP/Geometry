#!/bin/bash

# USEAGE: source job_caller.sh [task=save|compare]

cd /projappl/project_2001970/Geometry/logs

task=$1

if [ $task == 'compare' ]; do
  for modname in 'bert-base-cased' 'bert-base-uncased' 'opus-mt-en-af' 'opus-mt-en-bg' 'opus-mt-en-ca' 'opus-mt-en-cs' 'opus-mt-en-da' 'opus-mt-en-de' 'opus-mt-en-et' 'opus-mt-en-fi' 'opus-mt-en-fr' 'opus-mt-en-gl' 'opus-mt-en-hu' 'opus-mt-en-is' 'opus-mt-en-it' 'opus-mt-en-jap' 'opus-mt-en-nl' 'opus-mt-en-ro' 'opus-mt-en-sk' 'opus-mt-en-sv' 'opus-mt-en-uk' 'opus-mt-en-ru';do
    SAVEDATA=/scratch/project_2001970/Geometry/embeddings/attempt
    sbatch ../compare_embeddings.sh $modname $SAVEDATA
  done
fi

if [ $task == 'save' ];do
    sbatch ../save_embeddings.sh 

fi
