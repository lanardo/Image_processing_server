import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import glob
import sys

import logger

orb = cv2.ORB_create(nfeatures=500)
LEN_OF_GOODS = 10


def _calculate_location(img1, img2):

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    dx = 0.0
    dy = 0.0
    for m in matches[:LEN_OF_GOODS]:
        # print(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0], kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])
        dx += (kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0])/LEN_OF_GOODS
        dy += (kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])/LEN_OF_GOODS

    return int(dx), int(dy)


def _align_and_combine(data_path, output_path, size):
    # resizing the image with the source image size

    folder_out = output_path
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    # Scan all files on target folder
    cnt = 0
    logger.log_print("    Scan for align and resizing ...\n")
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    total = len(files)
    for f in files:

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.jpg':
            cnt += 1
            # logger.log_print("    {} / {}, File Name: {}\n" .format(cnt, total, f))
            sys.stdout.write("    {} / {}, File Name: {}\n".format(cnt, total, f))

            jpg_path = os.path.join(data_path, f)
            jpg_img = cv2.imread(jpg_path)

            if len(jpg_img.shape) != 3:  # Gray color image
                logger.log_print("    --- gray color --- \n")
                jpg_img = cv2.cvtcolor(jpg_img, cv2.COLOR_GRAY2RGB)

            h, w = jpg_img.shape[:2]

            # Resize the image with (size , size)
            new_jpg_img = cv2.resize(jpg_img, (size, int(size * h / w)))

            # create a combined image with size (size, 2*size)
            combined = np.zeros((size, size * 2, 3), dtype=np.uint8)

            # aligned_jpg_canvas[
            combined[int(size / 2 - size * h / w / 2):int(size / 2 - size * h / w / 2) + int(size * h / w), :size] \
                = new_jpg_img

            # save combined image in different folders
            new_combined_path = os.path.join(folder_out, fn + '.png')
            cv2.imwrite(new_combined_path, combined)

    return cnt


def align_combine(input_dir, output_dir):

    """
    mode="combine", size=256
        resize the images to be processed
        with size(size=256x256) and mode(mode="combined")

    Args:
        input_dir: path to folder containing images
        output_dir: path to folder containing combined/output image

    Returns:
        number of processed files
    """

    if input_dir is None or output_dir is None:
        raise Exception("    input_dir or ouput_dir not defined\n")

    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if len(input_paths) == 0:
        # raise Exception("    input_dir contains no .jpg image files")
        logger.log_print("    input_dir contains no .jpg image files\n")
        return len(input_paths)

    # align and combined for testing
    cnts = _align_and_combine(input_dir, output_dir, size=256)
    logger.log_print("    {} files Successfully aligned and combined.\n".format(cnts))

    return cnts
