from tqdm import tqdm

from common import *
from dataset.reader import multi_mask_to_color_overlay, multi_mask_to_contour_overlay, mask_to_inner_contour
from utility.file import read_list_from_file
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import pathlib

WHITE_BACKGROUND = [
    "8f94a80b95a881d0efdec36affc915dca9609f4cba8134c4a91b219d418778aa",
    "4217e25defac94ff465157d53f5a24b8a14045b763d8606ec4a97d71d99ee381",
    "7f38885521586fc6011bef1314a9fb2aa1e4935bd581b2991e1d963395eab770",
    "c395870ad9f5a3ae651b50efab9b20c3e6b9aea15d4c731eb34c0cf9e3800a72",
    "4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40",
    "76a372bfd3fad3ea30cb163b560e52607a8281f5b042484c3a0fc6d0aa5a7450",
    "1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df",
    "091944f1d2611c916b98c020bd066667e33f4639159b2a92407fe5a40788856d",
    "8d05fb18ee0cda107d56735cafa6197a31884e0a5092dc6d41760fb92ae23ab4",
    "1a11552569160f0b1ea10bedbd628ce6c14f29edec5092034c2309c556df833e",
    "08275a5b1c2dfcd739e8c4888a5ee2d29f83eccfa75185404ced1dc0866ea992",
    "5d58600efa0c2667ec85595bf456a54e2bd6e6e9a5c0dff42d807bc9fe2b822e",
    "54793624413c7d0e048173f7aeee85de3277f7e8d47c82e0a854fe43e879cd12",
    "5e263abff938acba1c0cff698261c7c00c23d7376e3ceacc3d5d4a655216b16d",
    "3594684b9ea0e16196f498815508f8d364d55fea2933a2e782122b6f00375d04",
    "2a1a294e21d76efd0399e4eb321b45f44f7510911acd92c988480195c5b4c812"
]


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
    newimage_dir = os.path.join(data_root, 'stag`e1_images', 'stage1_train')
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
    multimask_dir = os.path.join(data_root, 'stage1_images', 'fixed_multi_masks_watershred_debug')
    os.makedirs(multimask_dir, exist_ok=True)
    overlays_dir = os.path.join(data_root, 'stage1_images', 'fixed_overlays_watershred_debug')
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
                    instance_contour_overlay = mask_to_inner_contour(mask)
                    mask = watershred_post_proess(mask, instance_contour_overlay, 20)
                    multi_mask[np.where(mask > 128)] = instance_count
                    instance_count += 1
            else:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                instance_contour_overlay = mask_to_inner_contour(mask)
                mask = watershred_post_proess(mask, instance_contour_overlay, 20)
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

# Calculate the average size of the nuclei for watershred


def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr<2:
        mean_area = 1
        mean_radius = 1
    else:
        mean_are = int(itemfreq(labels)[1:1].mean())
        mean_radius = int(np.round(np.sqrt(mean_are) / np.pi))

    return mean_area, mean_radius

# Watershred to fillin holes, and get rid of small mask instance


def watershred_post_proess(mask, contour, threshold_small_mask_instance):
    m_b = mask > threshold_otsu(mask)
    c_b = contour > threshold_otsu(contour)
    # print(contour.shape)
    # print(c_b.shape)
    # print(c_b.shape)
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)
    non_zero_entry_x, _ = np.nonzero(m_)
    # print(non_zero_entry_x.shape)
    # plt.imshow(mask)
    # plt.show()
    # plt.imshow(m_)
    # plt.show()
    if non_zero_entry_x.shape[0] < threshold_small_mask_instance:
        # print("find it")
        return np.zeros(mask.shape)
    return m_*255


def make_annotation_final():

    print('generating training annotations')
    train_image_root_origin = '../../data/kaggle-dsbowl-2018-dataset-fixes-master/stage1_train'
    train_image_root = '../../data/stage1_images/2018-4-12_dataset/stage1_train'
    train_masks_root = '../../data/stage1_images/2018-4-12_dataset/stage1_train_masks'
    train_overlays_root = '../../data/stage1_images/2018-4-12_dataset/stage1_train_overlays'
    os.makedirs(train_image_root, exist_ok=True)
    os.makedirs(train_masks_root, exist_ok=True)
    os.makedirs(train_overlays_root, exist_ok=True)

    paths = pathlib.Path(train_image_root_origin).glob('*/images/*.png')
    for path in tqdm(paths):
        path_split = str(path).split('/')
        folder, name = path_split[-4], path_split[-3]
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, :3]
        if np.any(im[:, :, 0] != im[:, :, 1]):
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            im = im.astype(np.float)
            im = -im[:, :, np.newaxis].repeat(3, axis=2)
            im -= im.min()
            im /= im.max()
            im = im**1.5 * 255
        if name in WHITE_BACKGROUND:
            im = -im
        im = normalize(im.astype(np.float))
        masks_paths = pathlib.Path(train_image_root_origin).glob('%s/masks/*.png' % name)
        multi_mask = np.zeros((im.shape[0], im.shape[1]), np.int32)
        for k, p in enumerate(masks_paths):
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = k + 1

        # check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, im, [0, 255, 0])
        all = np.hstack((im, contour_overlay, color1_overlay,)).astype(np.uint8)

        cv2.imwrite(os.path.join(train_image_root, name + '.png'), im[:, :, 0])
        np.save(os.path.join(train_masks_root, name + '.npy'), multi_mask)
        cv2.imwrite(os.path.join(train_overlays_root, name + '.png'), all)

    print('generating validation annotations')
    valid_image_root_origin = '../../data/DSB2018_stage1_test-master/stage1_test'
    valid_image_root = '../../data/stage1_images/2018-4-12_dataset/stage1_valid'
    valid_masks_root = '../../data/stage1_images/2018-4-12_dataset/stage1_valid_masks'
    valid_overlays_root = '../../data/stage1_images/2018-4-12_dataset/stage1_valid_overlays'
    os.makedirs(valid_image_root, exist_ok=True)
    os.makedirs(valid_masks_root, exist_ok=True)
    os.makedirs(valid_overlays_root, exist_ok=True)

    paths = pathlib.Path(valid_image_root_origin).glob('*/images/*.png')
    for path in tqdm(paths):
        path_split = str(path).split('/')
        folder, name = path_split[-4], path_split[-3]
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, :3]
        if np.any(im[:, :, 0] != im[:, :, 1]):
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            im = im.astype(np.float)
            im = -im[:, :, np.newaxis].repeat(3, axis=2)
            im -= im.min()
            im /= im.max()
            im = im**1.5 * 255
        if name in WHITE_BACKGROUND:
            im = -im
        im = normalize(im.astype(np.float))
        masks_paths = pathlib.Path(valid_image_root_origin).glob('%s/masks/*.png' % name)
        multi_mask = np.zeros((im.shape[0], im.shape[1]), np.int32)
        for k, p in enumerate(masks_paths):
            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = k + 1

        # check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, im, [0, 255, 0])
        all = np.hstack((im, contour_overlay, color1_overlay,)).astype(np.uint8)

        cv2.imwrite(os.path.join(valid_image_root, name + '.png'), im[:, :, 0])
        np.save(os.path.join(valid_masks_root, name + '.npy'), multi_mask)
        cv2.imwrite(os.path.join(valid_overlays_root, name + '.png'), all)

    # print('generating external annotations')
    # external_image_root_origin = '../../data/2018-4-12_dataset/stage1_images/external'
    # external_image_root = '../../data/2018-4-12_dataset/stage1_images/external_gray'
    # external_masks_root = '../../data/2018-4-12_dataset/stage1_images/external_masks'
    # external_overlays_root = '../../data/2018-4-12_dataset/stage1_images/external_gray_overlays'
    # os.makedirs(external_image_root, exist_ok=True)
    # os.makedirs(external_overlays_root, exist_ok=True)
    #
    # paths = pathlib.Path(external_image_root_origin).glob('*.png')
    # for path in tqdm(paths):
    #     path_split = str(path).split('/')
    #     name = path_split[-1].strip('.png')
    #     im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    #     im = im.astype(np.float)
    #     im = -im[:, :, np.newaxis].repeat(3, axis=2)
    #     im -= im.min()
    #     im /= im.max()
    #     # im = im ** 1.5 * 255
    #     im = im * 255
    #     im = normalize(im.astype(np.float))
    #     multi_mask = np.load(external_masks_root + '/%s.npy' % name)
    #
    #     # check
    #     color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
    #     color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
    #     contour_overlay = multi_mask_to_contour_overlay(multi_mask, im, [0, 255, 0])
    #     all = np.hstack((im, contour_overlay, color1_overlay,)).astype(np.uint8)
    #
    #     cv2.imwrite(os.path.join(external_image_root, name + '.png'), im[:, :, 0])
    #     cv2.imwrite(os.path.join(external_overlays_root, name + '.png'), all)


def normalize(im):
    im -= im.min()
    im /= im.max()
    im *= 255
    return im


def make_test_annotation_final():

    print('generating testing annotations')
    test_image_root_origin = '../../data/stage2_test_final'
    test_image_root = '../../data/stage1_images/2018-4-12_dataset/stage2_test'
    os.makedirs(test_image_root, exist_ok=True)

    paths = pathlib.Path(test_image_root_origin).glob('*/images/*.png')
    for path in tqdm(paths):
        path_split = str(path).split('/')
        folder, name = path_split[-4], path_split[-3]
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)[:, :, :3]
        im = normalize(im.astype(np.float))
        cv2.imwrite(test_image_root + '/%s.png' % name, im)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_make_train_annotation()
    # run_make_test_annotation()

    # data_root = '../../data'

    # print(os.getcwd())
    # ids_fixed_masks = read_list_from_file(os.path.join(data_root, 'image_sets', 'fixed_mask_id.txt'), comment='#')
    # print(ids_fixed_masks)

    # run_make_train_annotation_fixing_masks(data_root, 'train1_ids_all_670')

    make_annotation_final()

    # make_test_annotation_final()

    print('sucess!')
