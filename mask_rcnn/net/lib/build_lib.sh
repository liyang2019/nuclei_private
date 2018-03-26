#!/bin/bash

ARCH=sm_37


# build the torch *.so lib for your system

echo '##########################################################'
echo '########### building torch NMS ###########################'
echo '##########################################################'
TORCH_NMS_DIR=box/nms/torch_nms/
nvcc -c -o $TORCH_NMS_DIR'src/nms_kernel.cu.o' $TORCH_NMS_DIR'src/nms_kernel.cu' -x cu -Xcompiler -fPIC -arch=$ARCH
python $TORCH_NMS_DIR'build.py'

echo '##########################################################'
echo '########### building roi align pooling layer #############'
echo '##########################################################'
ROI_ALIGN_POLL_TF_DIR=roi_align_pool_tf/
nvcc -c -o $ROI_ALIGN_POLL_TF_DIR'src/crop_and_resize_kernel.cu.o' $ROI_ALIGN_POLL_TF_DIR'src/crop_and_resize_kernel.cu' -x cu -Xcompiler -fPIC -arch=$ARCH
python roi_align_pool_tf/build.py

# build the cython *.so lib for your system

echo '##########################################################'
echo '########### building cython NMS ##########################'
echo '##########################################################'
cd box/nms/cython_nms
python setup.py build_ext --inplace

echo '##########################################################'
echo '########### building gpu NMS #############################'
echo '##########################################################'
cd ../gpu_nms/
python setup.py build_ext --inplace

echo '##########################################################'
echo '########### building cython overlap layer ################'
echo '##########################################################'
cd ../../overlap/cython_overlap
python setup.py build_ext --inplace
