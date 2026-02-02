# The default next_k is 4, reset next_k to 1 for traditional model training.

for pred_len in 96 192 336 720
do
    # Models are larger on traffic dataset, adjust test_k accordingly for limited GPU memory.
    training_k=4
    test_k=$((800 / ${pred_len}))

    ############################
    # iTransformer / Transformer
    ############################
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_96_${pred_len} \
      --model iTransformer \
      --data custom \
      --features M \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 64 --training_k ${training_k} --test_k ${test_k} \
      --itr 1

    ######################
    # iInformer / Informer
    ######################
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_96_${pred_len} \
      --model iInformer \
      --data custom \
      --features M \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --batch_size 64 --training_k ${training_k} --test_k ${test_k} \
      --itr 1
    
    ##########################
    # iFlowformer / Flowformer
    ##########################
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_96_${pred_len} \
      --model iFlowformer \
      --data custom \
      --features M \
      --pred_len ${pred_len} \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --batch_size 32 --training_k ${training_k} --test_k ${test_k} \
      --itr 1

    # ############################
    # # iFlashformer / Flashformer
    # ############################
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id traffic_96_${pred_len} \
      --model iFlashformer \
      --data custom \
      --features M \
      --pred_len ${pred_len} \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --batch_size 16 --training_k ${training_k} --test_k ${test_k} \
      --itr 1
done
