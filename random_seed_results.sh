# Commenting the following lines in run.py to fix the random seed for reproducibility in experiments.
# ```python
# fix_seed = 2023
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)
# ````
# 
# The default next_k is 4, reset next_k to 1 for traditional model training.

for pred_len in 96 192 336 720
do
    training_k=4
    test_k=$((1500 / ${pred_len}))

    ############################
    # iTransformer / Transformer
    ############################
    for itr in 1 2 3 4 5
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_${pred_len} \
        --model iTransformer \
        --data custom \
        --features M \
        --pred_len ${pred_len} \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 512 \
        --batch_size 512 --training_k ${training_k} --test_k ${test_k} \
        --itr ${itr}
    done

    ######################
    # iInformer / Informer
    ######################
    for itr in 1 2 3 4 5
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_${pred_len} \
        --model iInformer \
        --data custom \
        --features M \
        --pred_len ${pred_len} \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --batch_size 512 --training_k ${training_k} --test_k ${test_k} \
        --itr ${itr}
    done
    
    ##########################
    # iFlowformer / Flowformer
    ##########################
    for itr in 1 2 3 4 5
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_${pred_len} \
        --model iFlowformer \
        --data custom \
        --features M \
        --pred_len ${pred_len} \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --batch_size 512 --training_k ${training_k} --test_k ${test_k} \
        --itr ${itr}
    done

    ############################
    # iFlashformer / Flashformer
    ############################
    for itr in 1 2 3 4 5
    do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_${pred_len} \
        --model iFlashformer \
        --data custom \
        --features M \
        --pred_len ${pred_len} \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --batch_size 512 --training_k ${training_k} --test_k ${test_k} \
        --itr ${itr}
    done
done
