from common import *
from dataset.reader import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from utility.file import read_list_from_file


def run_make_test_annotation(data_root, image_set):

    imageset_file_loc = os.path.join(data_root, 'image_sets', image_set)

    # split = 'test1_ids_all_65'
    ids = read_list_from_file(imageset_file_loc, comment='#')
    newimage_dir = os.path.join(data_root, 'stage1_images', 'stage1_test')
    os.makedirs(newimage_dir, exist_ok=True)

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        print(id)
        folder, name = ids[i].split('/')
        image_loc = os.path.join(data_root, 'stage1_images', 'official', folder, name, 'images', '*.png')
        image_files = glob.glob(image_loc)
        assert (len(image_files) == 1)
        image_file = image_files[0]
        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(newimage_dir, name + '.png'), image)
        # image_show('image', image)
        # cv2.waitKey(1)


def run_make_train_annotation(data_root, image_set):
    imageset_file_loc = os.path.join(data_root, 'image_sets', image_set)
    # split = 'train1_ids_all_670'
    newimage_dir = os.path.join(data_root, 'stage1_images', 'stage1_train')
    os.makedirs(newimage_dir, exist_ok=True)
    multimask_dir = os.path.join(data_root, 'stage1_images', 'multi_masks')
    os.makedirs(multimask_dir, exist_ok=True)
    overlays_dir = os.path.join(data_root, 'stage1_images', 'overlays')
    os.makedirs(overlays_dir, exist_ok=True)

    ids = read_list_from_file(imageset_file_loc, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        print(id)
        name, folder = id.split('/')
        image_loc = os.path.join(data_root, 'stage1_images', 'official', folder, name, 'images', '*.png')
        image_files = glob.glob(image_loc)
        assert (len(image_files) == 1)
        image_file = image_files[0]
        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)
        masks_loc = os.path.join(data_root, 'stage1_images', 'official', folder, name, 'masks', '*.png')
        mask_files = glob.glob(masks_loc)
        mask_files.sort()
        num_masks = len(mask_files)
        for k in range(num_masks):
            mask_file = mask_files[k]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = k + 1

        # check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0, 255, 0])
        all = np.hstack((image, contour_overlay, color1_overlay,)).astype(np.uint8)

        cv2.imwrite(os.path.join(newimage_dir, name + '.png'), image)
        np.save(os.path.join(multimask_dir, name + '.npy'), multi_mask)
        cv2.imwrite(os.path.join(multimask_dir, name + '.png'), color_overlay)
        cv2.imwrite(os.path.join(overlays_dir, name + '.png'), all)


def run_make_train_annotation_fixing_masks(data_root, image_set):
    imageset_file_loc = os.path.join(data_root, 'image_sets', image_set)
    multimask_dir = os.path.join(data_root, 'stage1_images', 'fixed_multi_masks')
    os.makedirs(multimask_dir, exist_ok=True)
    overlays_dir = os.path.join(data_root, 'stage1_images', 'fixed_overlays')
    os.makedirs(overlays_dir, exist_ok=True)

    ids = read_list_from_file(imageset_file_loc, comment='#')

    fixed_masks_loc = os.path.join(data_root, 'stage1_images', 'fixed_mask_new', '*')
    fixed_masks_image_keys = [loc.split('/')[-1] for loc in glob.glob(fixed_masks_loc)]

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]
        print(id)
        folder, name = id.split('/')
        image_loc = os.path.join(data_root, 'stage1_images', 'official', folder, name, 'images', '*.png')
        image_files = glob.glob(image_loc)
        assert (len(image_files) == 1)
        image_file = image_files[0]
        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        fixed_masks_instance_keys = set()
        if name in fixed_masks_image_keys:
            fixed_masks_loc_instances = os.path.join(data_root, 'stage1_images', 'fixed_mask_new', name, '*')
            fixed_masks_instance_keys = [loc.split('/')[-1][:-7]
                                         for loc in glob.glob(fixed_masks_loc_instances)]
            fixed_masks_instance_keys = set(fixed_masks_instance_keys)

        print(fixed_masks_instance_keys)

        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)
        masks_loc = os.path.join(data_root, 'stage1_images', 'official', folder, name, 'masks', '*.png')
        mask_files = glob.glob(masks_loc)
        mask_files.sort()
        instance_count = 1
        for mask_file in mask_files:
            mask_key = mask_file.split('/')[-1].strip('.png')
            if mask_key in fixed_masks_instance_keys:
                fixed_mask_files = glob.glob(
                    os.path.join(data_root, 'stage1_images', 'fixed_mask_new', name, mask_key + '*.png'))
                print(fixed_mask_files)
                for fixed_mask_file in fixed_mask_files:
                    mask = cv2.imread(fixed_mask_file, cv2.IMREAD_GRAYSCALE)
                    multi_mask[np.where(mask > 128)] = instance_count
                    instance_count += 1
            else:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                multi_mask[np.where(mask > 128)] = instance_count
                instance_count += 1

        # check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0, 255, 0])
        all = np.hstack((image, contour_overlay, color1_overlay,)).astype(np.uint8)

        np.save(os.path.join(multimask_dir, name + '.npy'), multi_mask)
        cv2.imwrite(os.path.join(multimask_dir, name + '.png'), color_overlay)
        cv2.imwrite(os.path.join(overlays_dir, name + '.png'), all)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_make_train_annotation()
    # run_make_test_annotation()

    data_root = '../../data'

    # print(os.getcwd())
    # ids_fixed_masks = read_list_from_file(os.path.join(data_root, 'image_sets', 'fixed_mask_id.txt'), comment='#')
    # print(ids_fixed_masks)

    run_make_train_annotation_fixing_masks(data_root, 'train1_ids_all_670')

    print('sucess!')
