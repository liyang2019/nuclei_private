export PYTHONPATH=../:../mask_rcnn:../ensemble

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode none \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode hflip \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode vflip \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode scaleup \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode scaledown \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt

python predict.py \
    --initial_checkpoint ../results/2018-4-14_gray800_box/2018-04-15_03-47-56/checkpoint/00026500_model.pth \
    --test_augment_mode blur \
    --result_dir a020_2018-4-15_models_gray690_00037500_model \
    >> log.txt