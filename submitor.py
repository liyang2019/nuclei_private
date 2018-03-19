import imageio
import os
import numpy as np
import pandas as pd
from scipy import ndimage


class Submitor:

    def __init__(self, model, test_loader, output_dir, cuda):
        """
        Initialize a submitor for Kaggle. using a model to predict the segmentation, and
        save the result into a run length encoding csv file.
        :param model: A trained model.
        :param test_loader: The test data set loader.
        :param output_dir: An output directory to save the submission file.
        :param cuda: True is cuda available.
        """
        self.model = model
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.cuda = cuda

    def generate_submission_file(self, flag):
        """
        Generate a submission file for all image in a dataloader.
        """
        count = 0
        for img, _, (img_key,) in self.test_loader:
            print('analyzing image: ', img_key)
            count += 1
            all_df = pd.DataFrame()
            im_df = self.predict_one_image(img, img_key)
            all_df = all_df.append(im_df, ignore_index=True)
            all_df.to_csv(os.path.join(self.output_dir, flag + '_submission.csv'), index=None)
        assert count == 65

    def predict_one_image(self, img, img_key):
        img = img.cuda() if self.cuda else img
        seg = self.model.predict(img)
        seg = seg.squeeze()
        self.save_image(seg * 255, img_key)

        # Regenerate the labels
        labels, nlabels = ndimage.label(seg)
        print('There are {} separate components / objects detected.'.format(nlabels))

        # Loop through labels and add each to a DataFrame
        im_df = pd.DataFrame()
        for label_num in range(1, nlabels + 1):
            seg_instance = np.where(labels == label_num, 1, 0)
            if seg_instance.flatten().sum() > 20:
                rle = self.rle_encoding(seg_instance)
                s = pd.Series({'ImageId': img_key, 'EncodedPixels': rle})
                im_df = im_df.append(s, ignore_index=True)

        return im_df

    def save_image(self, img, img_key):
        imageio.imsave(os.path.join(self.output_dir, img_key), img)

    @staticmethod
    def rle_encoding(x):
        """
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        """
        dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if b > prev + 1:
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return " ".join([str(i) for i in run_lengths])
