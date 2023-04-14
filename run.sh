dataname=$1
n_class=18
opt_list='Adam'
lr_list='1e-3'
aug_list='flip'
bs_list='64'
model_list='efficientnet_b0'


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
                    exp_name="${model}_CE_${lr}_${bs}_${aug}_scheduler"
                    
                    if [ -d "$exp_name" ]
                    then
                        echo "$exp_name is exist"
                    else
                        python train.py \
                            --model $model \
                            --exp_name $exp_name \
                            --n_class $n_class \
                            --optimizer $opt \
                            --aug $aug \
                            --batch_size $bs \
                            --lr $lr \
                            --scheduler \
                            --epochs 50
                    fi

                    # # not use scheduler
                    # echo "model:$model, bs: $bs, opt: $opt, lr: $lr, aug: $aug, use_sched: False"
                    # exp_name="${model}_CE_${lr}_${bs}_${aug}"

                    # if [ -d "$exp_name" ]
                    # then
                    #     echo "$exp_name is exist"
                    # else
                    #     python train.py \
                    #         --model $model \
                    #         --exp_name $exp_name \
                    #         --n_class $n_class \
                    #         --optimizer $opt \
                    #         --aug $aug \
                    #         --batch_size $bs \
                    #         --lr $lr \
                    #         --epochs 50
                    # fi
                done
            done
        done
    done
done