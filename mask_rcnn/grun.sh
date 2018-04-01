python main.py \
       --learning_rate 0.002 \
       --input_width 128 --input_height 128 \
       --train_split train1_ids_gray2_500 \
       --val_split valid1_ids_gray2_43 \
       --batch_size 2 \
       --result_dir results/2018-3-31_gray500_size128 \
       --print_every 100 \
       --iter_accum 2 \
       --save_model_every 1000 \
       --iter_valid 100 \
       --is_validation \
#       --initial_checkpoint /home/li/nuclei_private/mask_rcnn/results/2018-3-29_purple_108/checkpoint/model_saved/00060500_model.pth

