dataname=$1
n_class=18
opt_list='Adam'
lr_list='0.001 0.0005'
aug_list='None'
bs_list='32 64'
model_list='efficientnet'


for model in $model_list
do
    for bs in $bs_list
    do
        for opt in $opt_list
        do
            for lr in $lr_list
            do
                for aug in $aug_list
                do
                    # use scheduler
                    echo "model:$model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: True"
                    exp_name="bs_$bs-opt_$opt-lr_$lr-aug_$aug-use_sched"
                    
                    if [ -d "$exp_name" ]
                    then
                        echo "$exp_name is exist"
                    else
                        python main.py \
                            --model $model \
                            --exp_name $exp_name \
                            --n_class $n_class \
                            --optimizer $opt \
                            --aug-name $aug \
                            --batch_size $bs \
                            --lr $lr \
                            --scheduler \
                            --epochs 50
                    fi

                    # not use scheduler
                    echo "model:$model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: False"
                    exp_name="bs_$bs-opt_$opt-lr_$lr-aug_$aug"

                    if [ -d "$exp_name" ]
                    then
                        echo "$exp_name is exist"
                    else
                        python main.py \
                            --model $model \
                            --exp_name $exp_name \
                            --n_class $n_class \
                            --optimizer $opt \
                            --aug-name $aug \
                            --batch_size $bs \
                            --lr $lr \
                            --scheduler \
                            --epochs 50
                    fi
                done
            done
        done
    done
done