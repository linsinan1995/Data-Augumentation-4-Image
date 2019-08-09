#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @author: Lin Sinan
# @contact: mynameisxiaou@gmail.com
# @github: linsinan1995
# @file: main.py
# @time: 2019/8/8 下午3:22
# @desc: augumentation tool for face landmarks
#


from config import *
import os.path as osp
import numpy as np
import cv2
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa

def check_ld_boundary(landmark, shape):
    """
    :param landmark: [196,] landmark
    :param shape:  image.shape
    :return:  bool
    """
    x = shape[0]
    y = shape[1]
    xs = landmark[::2]
    ys = landmark[1::2]
    return sum(xs < 0) + sum(xs-x > 0) > THRED or sum(ys < 0) + sum(ys-y > 0) > THRED

def genData(isTrain = True):

    if isTrain:
        data_landmarks = np.loadtxt(train_landmarks_path, usecols=([i for i in range(NUM_LANDMARKS*2)]), dtype = np.float)
        data_faceArea = np.loadtxt(train_landmarks_path, usecols=([NUM_LANDMARKS*2+i for i in range(4)]), dtype = np.float)
        data_image = np.loadtxt(train_landmarks_path, usecols=(-1), dtype = np.str)
    else:
        data_landmarks = np.loadtxt(test_landmarks_path, usecols=([i for i in range(NUM_LANDMARKS*2)]), dtype = np.float)
        data_faceArea = np.loadtxt(test_landmarks_path, usecols=([NUM_LANDMARKS*2+i for i in range(4)]), dtype = np.float)
        data_image = np.loadtxt(test_landmarks_path, usecols=(-1), dtype = np.str)

    f = open(landmark_path_for_save)
    #todo:augumentation
    # https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B01%20-%20Augment%20Keypoints.ipynb
    for _i in range(1):#len(train_data_image)):
        IND = _i
        sometimes = lambda aug: iaa.Sometimes(0.4, aug)
        sometimes_01 = lambda aug: iaa.Sometimes(0.18, aug)

        # load pic, add a new dim and stack 20 of it together
        image = cv2.imread(osp.join(img_path,data_image[IND]))

        cols = data_faceArea[IND][0]-10 if data_faceArea[IND][0]-10 > 0 else 0
        rows = data_faceArea[IND][1]-10 if data_faceArea[IND][1]-10 > 0 else 0
        weight = data_faceArea[IND][2]+10 if data_faceArea[IND][2]+10 < image.shape[1] else image.shape[1]
        height = data_faceArea[IND][3]+10 if data_faceArea[IND][3]+10 < image.shape[0] else image.shape[0]
        print(cols, rows, weight, height)
        image = image[int(rows):int(height), int(cols):int(weight), :]
        # images = np.concatenate((
        #     [np.expand_dims(image, axis=0)] * 20
        # ), dtype=np.uint8)

        # landmarks
        kpsoi = KeypointsOnImage([
            Keypoint(x=data_landmarks[IND][i]-cols, y=data_landmarks[IND][i+1]-rows) for i in range(0, NUM_LANDMARKS*2, 2)
        ], shape=image.shape)

        # kpsois = [kpsoi.to_xy_array()]*20

        seq = iaa.Sequential([
            iaa.Fliplr(p=0.35),
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-15, 15), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            )),
            iaa.AddToHueAndSaturation((-25, 25)),
            iaa.OneOf([
                iaa.Multiply((0.5, 1.5)),
                iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.5, 1.5)),
                    second=iaa.ContrastNormalization((0.5, 2.0))
                )
            ]),
            sometimes_01(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            sometimes_01(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        ])

        for index in range(24):
            image_aug, kpsoi_aug = seq(image=image, keypoints=kpsoi)


            ld = kpsoi_aug.to_xy_array().reshape(-1)
            if check_ld_boundary(ld, image_aug.shape):
                print("PASS!")
                continue

            _path = osp.join(train_path_for_save, str(index)+"_"+ data_image[IND][
                                                                  data_image[IND].rfind("/")+1:]
                             )

            # cv2.imshow(
            #     "image",
            #     np.hstack([
            #         kpsoi.draw_on_image(image, size=7),
            #         kpsoi_aug.draw_on_image(image_aug, size=7)
            #     ])
            # )


            # for ind in range(0, NUM_LANDMARKS*2, 2):
            #     cv2.circle(image_aug, (ld[ind], ld[ind+1]), 1, (76, 201, 255), 1)
            # cv2.imshow("img", image_aug)
            # cv2.waitKey(0)

            cv2.imwrite(_path, image_aug)

        # VISUALIZATION:
        #
        # cv2.imwrite(
        #     "image.jpg",
        #     np.vstack([
        #         np.hstack([
        #             kpsoi.draw_on_image(image, size=7),
        #             kpsoi_augs[0].draw_on_image(image_augs[0], size=7),
        #             kpsoi_augs[1].draw_on_image(image_augs[1], size=7)
        #         ]),
        #         np.hstack([
        #             kpsoi_augs[2].draw_on_image(image_augs[2], size=7),
        #             kpsoi_augs[3].draw_on_image(image_augs[3], size=7),
        #             kpsoi_augs[4].draw_on_image(image_augs[4], size=7)
        #         ]),
        #         np.hstack([
        #             kpsoi_augs[5].draw_on_image(image_augs[5], size=7),
        #             kpsoi_augs[6].draw_on_image(image_augs[6], size=7),
        #             kpsoi_augs[7].draw_on_image(image_augs[7], size=7)
        #         ])
        #     ])
        # )


if __name__ == "__main__":
    genData(True)