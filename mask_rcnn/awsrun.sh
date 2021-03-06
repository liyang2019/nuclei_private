python main.py \
       --data_dir ../data/2018-4-12_dataset \
       --train_split ids_train800 \
       --valid_split ids_valid49 \
       --visualize_split ids_visualize \
       --image_folder_train stage1_tot \
       --masks_folder_train stage1_tot_masks \
       --image_folder_valid stage1_tot \
       --masks_folder_valid stage1_tot_masks \
       --image_folder_visualize visualize \
       --masks_folder_visualize visualize_masks \
       --result_dir ../results/2018-4-14_gray800_box \
       --input_width 128 --input_height 128 \
       --learning_rate 0.01 \
       --batch_size 8 \
       --iter_accum 2 \
       --print_every 10 \
       --iter_valid 200 \
       --save_model_every 500 \
       --is_validation \
       --num_iters 18000 \
       --masknet 4conv \
       --color_scheme gray \
       --feature_channels 256 \
       --train_box_only \
       --run \
#       --initial_checkpoint /home/ubuntu/nuclei_private/results/2018-4-14_gray800_box/latest_model.pth \


python main.py \
       --data_dir ../data/2018-4-12_dataset \
       --train_split ids_train800 \
       --valid_split ids_valid49 \
       --visualize_split ids_visualize \
       --image_folder_train stage1_tot \
       --masks_folder_train stage1_tot_masks \
       --image_folder_valid stage1_tot \
       --masks_folder_valid stage1_tot_masks \
       --image_folder_visualize visualize \
       --masks_folder_visualize visualize_masks \
       --result_dir ../results/2018-4-14_gray800_box \
       --input_width 256 --input_height 256 \
       --learning_rate 0.01 \
       --batch_size 7 \
       --iter_accum 2 \
       --print_every 10 \
       --iter_valid 200 \
       --save_model_every 500 \
       --is_validation \
       --num_iters 18000 \
       --masknet 4conv \
       --color_scheme gray \
       --feature_channels 256 \
       --run \
       --initial_checkpoint /home/ubuntu/nuclei_private/results/2018-4-14_gray800_box/latest_model.pth \