python main.py \
       --data_dir ../data/2018-4-12_dataset \
       --train_split ids_train \
       --valid_split ids_valid \
       --visualize_split ids_visualize \
       --image_folder_train stage1_train \
       --masks_folder_train stage1_train_masks \
       --image_folder_valid stage1_valid \
       --masks_folder_valid stage1_valid_masks \
       --image_folder_visualize visualize \
       --masks_folder_visualize visualize_masks \
       --result_dir ../results/2018-4-13_gray690 \
       --input_width 128 --input_height 128 \
       --learning_rate 0.01 \
       --batch_size 8 \
       --iter_accum 1 \
       --print_every 10 \
       --iter_valid 10 \
       --is_validation \
       --num_iters 20 \
       --masknet 4conv \
       ----color_scheme gray \
       --feature_channels 128 \
       --train_box_only

python main.py \
       --data_dir ../data/2018-4-12_dataset \
       --train_split ids_train \
       --valid_split ids_valid \
       --visualize_split ids_visualize \
       --image_folder_train stage1_train \
       --masks_folder_train stage1_train_masks \
       --image_folder_valid stage1_valid \
       --masks_folder_valid stage1_valid_masks \
       --image_folder_visualize visualize \
       --masks_folder_visualize visualize_masks \
       --result_dir ../results/2018-4-13_gray690 \
       --input_width 128 --input_height 128 \
       --learning_rate 0.01 \
       --batch_size 8 \
       --iter_accum 1 \
       --print_every 10 \
       --iter_valid 10 \
       --is_validation \
       --num_iters 20 \
       --masknet 4conv \
       ----color_scheme gray \
       --feature_channels 128 \


#       --initial_checkpoint /home/li/nuclei_private/mask_rcnn/results/2018-3-31_gray500_size128/checkpoint/model_saved_mask/00042500_model.pth \
