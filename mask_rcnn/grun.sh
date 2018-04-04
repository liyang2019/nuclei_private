python main.py \
       --learning_rate 0.01 \
       --input_width 128 --input_height 128 \
       --train_split disk0_ids_dummy_9 \
       --valid_split valid1_ids_gray2_43 \
       --batch_size 4 \
       --result_dir results/2018-3-31_gray500_size128 \
       --print_every 10 \
       --iter_accum 1 \
       --iter_valid 100 \
#       --initial_checkpoint /home/li/nuclei_private/mask_rcnn/results/2018-3-31_gray500_size128/checkpoint/model_saved_mask/00042500_model.pth
#       --is_validation \