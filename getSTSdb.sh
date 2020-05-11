

MOSES=$1

#STS datasets: STS12, STS13, STS14, STS15, STS16
mkdir -p  data/STS

declare -A STS_tasks
declare -A STS_paths
declare -A STS_subdirs

STS_tasks=(["STS12"]="MSRpar MSRvid SMTeuroparl surprise.OnWN surprise.SMTnews" ["STS13"]="FNWN headlines OnWN" ["STS14"]="deft-forum deft-news headlines OnWN images tweet-news" ["STS15"]="answers-forums answers-students belief headlines images" ["STS16"]="answer-answer headlines plagiarism postediting question-question")

STS_paths=(["STS12"]="http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip" ["STS13"]="http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip" ["STS14"]="http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip" ["STS15"]="http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip"
["STS16"]="http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip")

STS_subdirs=(["STS12"]="test-gold" ["STS13"]="test-gs" ["STS14"]="sts-en-test-gs-2014" ["STS15"]="test_evaluation_task2a" ["STS16"]="sts2016-english-with-gs-v1.0")


MOSESTOKENIZER=$MOSES/tokenizer/tokenizer.perl

for task in "${!STS_tasks[@]}";
do
    fpath=${STS_paths[$task]}
    echo $fpath
    curl -Lo data/STS/data_$task.zip $fpath
    unzip data/STS/data_$task.zip -d data/STS
    mv data/STS/${STS_subdirs[$task]} data/STS/$task-en-test
    rm data/STS/data_$task.zip

    for sts_task in ${STS_tasks[$task]}
    do
        fname=STS.input.$sts_task.txt
        task_path=data/STS/$task-en-test/

        if [ "$task" = "STS16" ] ; then
            echo 'Handling STS2016'
            mv $task_path/STS2016.input.$sts_task.txt $task_path/$fname
            mv $task_path/STS2016.gs.$sts_task.txt $task_path/STS.gs.$sts_task.txt
        fi



        cut -f1 $task_path/$fname | $MOSESTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp1
        cut -f2 $task_path/$fname | $MOSESTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp2
        paste $task_path/tmp1 $task_path/tmp2 > $task_path/$fname
        rm $task_path/tmp1 $task_path/tmp2
    done

done
