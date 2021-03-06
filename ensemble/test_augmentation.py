from dataset.reader import *
from mask_rcnn.submit import run_npy_to_sumbit_csv, filter_small
from net.metric import run_length_decode, run_length_encode

sys.path.append(os.path.dirname(__file__))


# ensemble =======================================================


class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members = []
        self.center = {}

    def add_item(self, box, score, instance):
        if not self.center:
            self.members = [{
                'box': box, 'score': score, 'instance': instance
            }, ]
            self.center = {
                'box': box, 'score': score, 'union': (instance > 0.5), 'inter': (instance > 0.5),
            }
        else:
            self.members.append({
                'box': box, 'score': score, 'instance': instance
            })
            center_box = self.center['box'].copy()
            center_score = self.center['score']
            center_union = self.center['union'].copy()
            center_inter = self.center['inter'].copy()

            self.center['box'] = [
                min(box[0], center_box[0]),
                min(box[1], center_box[1]),
                max(box[2], center_box[2]),
                max(box[3], center_box[3]),
            ]
            self.center['score'] = max(score, center_score)
            self.center['union'] = center_union | (instance > 0.5)
            self.center['inter'] = center_inter & (instance > 0.5)

    def distance(self, box, score, instance):
        center_box = self.center['box']
        center_union = self.center['union']
        # center_inter = self.center['inter']

        x0 = int(max(box[0], center_box[0]))
        y0 = int(max(box[1], center_box[1]))
        x1 = int(min(box[2], center_box[2]))
        y1 = int(min(box[3], center_box[3]))

        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        box_intersection = w * h
        if box_intersection < 0.01:
            return 0

        x0 = int(min(box[0], center_box[0]))
        y0 = int(min(box[1], center_box[1]))
        x1 = int(max(box[2], center_box[2]))
        y1 = int(max(box[3], center_box[3]))

        i0 = center_union[y0:y1, x0:x1]  # center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1, x0:x1] > 0.5

        intersection = np.logical_and(i0, i1).sum()
        area = np.logical_or(i0, i1).sum()
        overlap = intersection / (area + 1e-12)

        return overlap


def do_clustering(boxes, scores, instances, threshold=0.5):
    clusters = []
    num_arguments = len(instances)
    for n in range(0, num_arguments):
        box = boxes[n]
        score = scores[n]
        instance = instances[n]

        num = len(instance)
        for m in range(num):
            b, s, i = box[m], score[m], instance[m]

            is_group = 0
            for c in clusters:
                iou = c.distance(b, s, i)

                if iou > threshold:
                    c.add_item(b, s, i)
                    is_group = 1

            if is_group == 0:
                c = Cluster()
                c.add_item(b, s, i)
                clusters.append(c)

    return clusters


def mask_to_more(mask):
    H, W = mask.shape[:2]
    box = []
    score = []
    instance = []

    for i in range(mask.max()):
        m = (mask == (i + 1))

        # filter by size, boundary, etc ....
        if m.sum() > 0:
            # box
            y, x = np.where(m)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            b = [x0, y0, x1, y1]

            # score
            s = 1

            # add --------------------
            box.append(b)
            score.append(s)
            instance.append(m)

            # image_show('m',m*255)
            # cv2.waitKey(0)

    box = np.array(box, np.float32)
    score = np.array(score, np.float32)
    instance = np.array(instance, np.float32)

    if len(box) == 0:
        box = np.zeros((0, 4), np.float32)
        score = np.zeros((0, 1), np.float32)
        instance = np.zeros((0, H, W), np.float32)

    return box, score, instance


def run_ensemble(data_dir, out_dir, ensemble_dirs):

    ## setup  --------------------------
    os.makedirs(out_dir + '/average_semantic_mask', exist_ok=True)
    os.makedirs(out_dir + '/cluster_union_mask', exist_ok=True)
    os.makedirs(out_dir + '/cluster_inter_mask', exist_ok=True)
    os.makedirs(out_dir + '/ensemble_mask', exist_ok=True)
    os.makedirs(out_dir + '/ensemble_mask_overlays', exist_ok=True)
    os.makedirs(out_dir + '/npys', exist_ok=True)

    names = glob.glob(os.path.join(ensemble_dirs[0], '*.npy'))
    names = [n.split('/')[-1].strip('.npy') for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for name in names:
        # name='1cdbfee1951356e7b0a215073828695fe1ead5f8b1add119b6645d2fdc8d844e'
        print(name)
        boxes = []
        scores = []
        instances = []

        average_semantic_mask = None
        for dir in ensemble_dirs:
            npy_file = os.path.join(dir, '%s.npy' % name)
            mask = np.load(npy_file).astype(np.int32)

            # im = cv2.imread(dir + '/overlays/%s/%s.mask.png' % (name, name), cv2.IMREAD_COLOR)
            # im = im[:, :, 0] * 256**2 + im[:, :, 1] * 256 + im[:, :, 2]
            # mask, _ = ndimage.label(im)

            if average_semantic_mask is None:
                average_semantic_mask = (mask > 0).astype(np.float32)
            else:
                average_semantic_mask = average_semantic_mask + (mask > 0).astype(np.float32)

            box, score, instance = mask_to_more(mask)
            boxes.append(box)
            scores.append(score)
            instances.append(instance)

        clusters = do_clustering(boxes, scores, instances, threshold=0.3)
        H, W = average_semantic_mask.shape[:2]

        # <todo> do your ensemble  here! =======================================
        ensemble_mask = np.zeros((H, W), np.int32)
        for i, c in enumerate(clusters):
            num_members = len(c.members)
            average = np.zeros((H, W), np.float32)  # e.g. use average
            for n in range(num_members):
                average = average + c.members[n]['instance']
            average = average / num_members

            ensemble_mask[average > 0.5] = i + 1
        np.save(out_dir + '/npys/%s.npy' % name, ensemble_mask)

        # do some post processing here ---
        # e.g. fill holes
        #      remove small fragment
        #      remove boundary
        # <todo> do your ensemble  here! =======================================

        # show clustering/ensemble results
        cluster_inter_mask = np.zeros((H, W), np.int32)
        cluster_union_mask = np.zeros((H, W), np.int32)
        for i, c in enumerate(clusters):
            cluster_inter_mask[c.center['inter']] = i + 1
            cluster_union_mask[c.center['union']] = i + 1

        color_overlay0 = multi_mask_to_color_overlay(cluster_inter_mask)
        color_overlay1 = multi_mask_to_color_overlay(cluster_union_mask)
        color_overlay2 = multi_mask_to_color_overlay(ensemble_mask)
        ##-------------------------
        average_semantic_mask = (average_semantic_mask / num_ensemble * 255).astype(np.uint8)
        average_semantic_mask = cv2.cvtColor(average_semantic_mask, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(out_dir + '/average_semantic_mask/%s.png' % (name,), average_semantic_mask)
        cv2.imwrite(out_dir + '/cluster_inter_mask/%s.mask.png' % (name,), color_overlay0)
        cv2.imwrite(out_dir + '/cluster_union_mask/%s.mask.png' % (name,), color_overlay1)
        cv2.imwrite(out_dir + '/ensemble_mask/%s.mask.png' % (name,), color_overlay2)

        # image_show('average_semantic_mask', average_semantic_mask)
        # image_show('cluster_inter_mask', color_overlay0)
        # image_show('cluster_union_mask', color_overlay1)
        # image_show('ensemble_mask',color_overlay2)

        if 1:
            folder = 'stage2_test'
            image = cv2.imread(data_dir + '/%s/%s.png' % (folder, name), cv2.IMREAD_COLOR)

            mask = ensemble_mask
            # norm_image = adjust_gamma(image, 2.5)
            norm_image = image
            color_overlay = multi_mask_to_color_overlay(mask)
            color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay)
            contour_overlay = multi_mask_to_contour_overlay(mask, norm_image, [0, 255, 0])
            all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
            # image_show('ensemble_mask', all)

            # psd
            cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s.png' % (name,), all)
            os.makedirs(out_dir + '/ensemble_mask_overlays/%s' % (name,), exist_ok=True)
            cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.png' % (name, name), image)
            cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.mask.png' % (name, name), color_overlay)
            cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.contour.png' % (name, name), contour_overlay)

        # pass


def rles_to_multimask(rles, H, W):
    multi_mask = np.zeros((H, W), dtype=np.int)
    for k, rle in enumerate(rles):
        mask = run_length_decode(rle, H, W) > 128
        multi_mask[mask] = k + 1
    return multi_mask


def run_ensemble_from_csvs(data_dir, out_dir, csv_file, names, name_to_rles_list):

    ## setup  --------------------------
    # os.makedirs(out_dir + '/average_semantic_mask', exist_ok=True)
    # os.makedirs(out_dir + '/cluster_union_mask', exist_ok=True)
    # os.makedirs(out_dir + '/cluster_inter_mask', exist_ok=True)
    # os.makedirs(out_dir + '/ensemble_mask', exist_ok=True)
    os.makedirs(out_dir + '/ensemble_mask_overlays', exist_ok=True)
    # os.makedirs(out_dir + '/npys', exist_ok=True)

    sorted(names)

    cvs_ImageId = []
    cvs_EncodedPixels = []

    # num_ensemble = len(name_to_rles_list)
    for index, name in enumerate(names):
        print(index, name)
        im = cv2.imread('../data/2018-4-12_dataset/stage2_test/' + name + '.png')
        H, W = im.shape[0], im.shape[1]

        boxes = []
        scores = []
        instances = []

        average_semantic_mask = None
        for name_to_rles in name_to_rles_list:
            rles = name_to_rles[name]
            mask = rles_to_multimask(rles, H, W)

            if average_semantic_mask is None:
                average_semantic_mask = (mask > 0).astype(np.float32)
            else:
                average_semantic_mask = average_semantic_mask + (mask > 0).astype(np.float32)

            box, score, instance = mask_to_more(mask)
            boxes.append(box)
            scores.append(score)
            instances.append(instance)

        clusters = do_clustering(boxes, scores, instances, threshold=0.3)

        # <todo> do your ensemble  here! =======================================
        ensemble_mask = np.zeros((H, W), np.int32)
        for i, c in enumerate(clusters):
            num_members = len(c.members)
            average = np.zeros((H, W), np.float32)  # e.g. use average
            for n in range(num_members):
                average = average + c.members[n]['instance']
            average = average / num_members
            ensemble_mask[average > 0.5] = i + 1

        multi_mask = filter_small(ensemble_mask, 8)
        num = int(multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask == m + 1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)
        # np.save(out_dir + '/npys/%s.npy' % name, ensemble_mask)

        # do some post processing here ---
        # e.g. fill holes
        #      remove small fragment
        #      remove boundary
        # <todo> do your ensemble  here! =======================================

        # show clustering/ensemble results
        # cluster_inter_mask = np.zeros((H, W), np.int32)
        # cluster_union_mask = np.zeros((H, W), np.int32)
        # for i, c in enumerate(clusters):
        #     cluster_inter_mask[c.center['inter']] = i + 1
        #     cluster_union_mask[c.center['union']] = i + 1

        # color_overlay0 = multi_mask_to_color_overlay(cluster_inter_mask)
        # color_overlay1 = multi_mask_to_color_overlay(cluster_union_mask)
        # color_overlay2 = multi_mask_to_color_overlay(ensemble_mask)
        ##-------------------------
        # average_semantic_mask = (average_semantic_mask / num_ensemble * 255).astype(np.uint8)
        # average_semantic_mask = cv2.cvtColor(average_semantic_mask, cv2.COLOR_GRAY2BGR)

        # cv2.imwrite(out_dir + '/average_semantic_mask/%s.png' % (name,), average_semantic_mask)
        # cv2.imwrite(out_dir + '/cluster_inter_mask/%s.mask.png' % (name,), color_overlay0)
        # cv2.imwrite(out_dir + '/cluster_union_mask/%s.mask.png' % (name,), color_overlay1)
        # cv2.imwrite(out_dir + '/ensemble_mask/%s.mask.png' % (name,), color_overlay2)

        # image_show('average_semantic_mask', average_semantic_mask)
        # image_show('cluster_inter_mask', color_overlay0)
        # image_show('cluster_union_mask', color_overlay1)
        # image_show('ensemble_mask',color_overlay2)

        if 1:
            folder = 'stage2_test'
            image = cv2.imread(data_dir + '/%s/%s.png' % (folder, name), cv2.IMREAD_COLOR)

            mask = ensemble_mask
            # norm_image = adjust_gamma(image, 2.5)
            norm_image = image
            color_overlay = multi_mask_to_color_overlay(mask)
            color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay)
            contour_overlay = multi_mask_to_contour_overlay(mask, norm_image, [0, 255, 0])
            all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
            # image_show('ensemble_mask', all)

            # psd
            cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s.png' % (name,), all)
            # os.makedirs(out_dir + '/ensemble_mask_overlays/%s' % (name,), exist_ok=True)
            # cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.png' % (name, name), image)
            # cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.mask.png' % (name, name), color_overlay)
            # cv2.imwrite(out_dir + '/ensemble_mask_overlays/%s/%s.contour.png' % (name, name), contour_overlay)

    df = pd.DataFrame({'ImageId': cvs_ImageId, 'EncodedPixels': cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])
    return cvs_ImageId, cvs_EncodedPixels


def main():

    # gcp
    data_dir = '../data/2018-4-12_dataset'
    out_root = 'ensemble'
    out_dir = os.path.join(out_root, 'output')
    ensemble_dirs = [
        # different predictors, test augments, etc ...
        os.path.join(out_root, 'none', 'submit', 'npys'),
        os.path.join(out_root, 'vflip', 'submit', 'npys'),
        os.path.join(out_root, 'hflip', 'submit', 'npys'),
        os.path.join(out_root, 'scaleup', 'submit', 'npys'),
        os.path.join(out_root, 'scaledown', 'submit', 'npys'),
    ]
    run_ensemble(data_dir, out_dir, ensemble_dirs)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # main()
    output_root = '2018-04-08_00-45-40_mask_rcnn_purple-yellow'
    npy_dir = output_root + '/output/npys'
    os.makedirs(output_root + '/submit', exist_ok=True)
    csv_file = output_root + '/submit/submit.csv'
    run_npy_to_sumbit_csv(npy_dir, csv_file)
    print('\nsucess!')
