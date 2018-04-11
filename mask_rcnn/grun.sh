python main.py \
       --learning_rate 0.01 \
       --input_width 128 --input_height 128 \
       --train_split train1_ids_gray2_500 \
       --valid_split valid1_ids_gray2_43 \
       --batch_size 8 \
       --result_dir results/2018-4-10_mini-unet_leaky \
       --print_every 10 \
       --iter_accum 1 \
       --iter_valid 100 \
       --image_folder_train stage1_train \
       --masks_folder_train fixed_multi_masks \
       --image_folder_valid stage1_train \
       --masks_folder_valid fixed_multi_masks \
       --is_validation \
       --num_iters 20 \
       --masknet mini-unet \
       --feature_channels 128

python main.py \
       --learning_rate 0.01 \
       --input_width 128 --input_height 128 \
       --train_split train1_ids_gray2_500 \
       --valid_split valid1_ids_gray2_43 \
       --batch_size 8 \
       --result_dir results/2018-4-10_mini-res_leaky \
       --print_every 10 \
       --iter_accum 1 \
       --iter_valid 100 \
       --image_folder_train stage1_train \
       --masks_folder_train fixed_multi_masks \
       --image_folder_valid stage1_train \
       --masks_folder_valid fixed_multi_masks \
       --is_validation \
       --num_iters 20 \
       --masknet mini-res \
       --feature_channels 128

python main.py \
       --learning_rate 0.01 \
       --input_width 128 --input_height 128 \
       --train_split train1_ids_gray2_500 \
       --valid_split valid1_ids_gray2_43 \
       --batch_size 8 \
       --result_dir results/2018-4-10_4conv_leaky \
       --print_every 10 \
       --iter_accum 1 \
       --iter_valid 100 \
       --image_folder_train stage1_train \
       --masks_folder_train fixed_multi_masks \
       --image_folder_valid stage1_train \
       --masks_folder_valid fixed_multi_masks \
       --is_validation \
       --num_iters 20 \
       --masknet 4conv \
       --feature_channels 128


#       --initial_checkpoint /home/li/nuclei_private/mask_rcnn/results/2018-3-31_gray500_size128/checkpoint/model_saved_mask/00042500_model.pth \
#       --is_validation \
#       --image_folder_train disk0/images \
#       --masks_folder_train disk0/multi_masks \
#       --image_folder_valid disk0/images \
#       --masks_folder_valid disk0/multi_masks \
#       --train_split disk0_ids_dummy_9 \
#       --valid_split disk0_ids_dummy_3 \