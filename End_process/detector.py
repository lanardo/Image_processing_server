import glob
import math
import os
import sys
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from .manager_s3 import image_upload_to_s3

from PIL import Image
from pytesseract import image_to_string

import logger
size = 256
KERNEL_1 = np.ones((5, 5), np.uint8)
KERNEL_2 = np.ones((5, 5), np.uint8)
KERNEL_3 = (11, 11)
HIST_RODS = 8
GRAD_MAX = 50
GRAD_SCALE = 1.0

RECT = "vehicle_points"
COLOR = "vehicle_color"
TXT = "image_text"
CLASS = "image_classification"
URL = "image_url"


def _img_proc(ori_im, rm_thm):

    """ Extract the mask image for car with the original image size """
    rm_thm_gray = cv2.cvtColor(rm_thm, cv2.COLOR_RGB2GRAY)
    _, rm_thm_bin = cv2.threshold(rm_thm_gray, 10, 255, cv2.THRESH_BINARY)
    # soil erosion and dilation, opposite of erosion
    rm_thm_bin = cv2.erode(rm_thm_bin, KERNEL_1, iterations=1)
    rm_thm_bin = cv2.dilate(rm_thm_bin, KERNEL_1, iterations=1)

    # resizing to fit the original size
    ori_h, ori_w, chs = ori_im.shape

    th_h, th_w = rm_thm_bin.shape[:2]

    M = np.float32(
        [[float(ori_w) / th_w, 0, 0], [0, float(ori_w) / th_w, int((ori_h - float(th_h * ori_w) / th_w) / 2)]])
    obj_mask = cv2.warpAffine(rm_thm_bin, M, (ori_w, ori_h))

    # obj_mask = cv2.erode(obj_mask, KERNEL_2, iterations=1)
    obj_mask = cv2.GaussianBlur(obj_mask, KERNEL_3, 5.0)
    obj_mask = cv2.threshold(obj_mask, 170, 255, cv2.THRESH_BINARY)[1]

    """ Color Recognition with historgram analysis """
    # color detection
    hist_b = cv2.calcHist([ori_im], [0], obj_mask, [HIST_RODS], [0, 256])
    hist_g = cv2.calcHist([ori_im], [1], obj_mask, [HIST_RODS], [0, 256])
    hist_r = cv2.calcHist([ori_im], [2], obj_mask, [HIST_RODS], [0, 256])

    b = np.argmax(hist_b) * 255 / HIST_RODS + 0.5 * 255 / HIST_RODS
    g = np.argmax(hist_g) * 255 / HIST_RODS + 0.5 * 255 / HIST_RODS
    r = np.argmax(hist_r) * 255 / HIST_RODS + 0.5 * 255 / HIST_RODS

    color = (int(r), int(g), int(b))

    """ Extract the rect of object from the Contours """
    # get contour from mask image
    _, contours, hierarchy = cv2.findContours(obj_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # merging contours
    rect = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        rect.append([x, y, x+w, y+h])
    rect = np.array(rect)
    pick = non_max_suppression(rect, probs=None, overlapThresh=0.65)

    max_square = 0.0
    rect = []
    for (x1, y1, x2, y2) in pick:
        square = math.fabs((x1 - x2) * (y1 - y2))

        if max_square < square:
            max_square = square
            rect = [(x1, y1), (x2, y2)]

    """ Combind the extracted object and the gradiented background """
    # the gradient mask for background image
    rm_im = np.zeros((ori_h, ori_w, 4), dtype=np.uint8)

    bk_mask = np.ones((ori_h, ori_w), dtype=np.float)
    for i in range(ori_h):
        bk_mask[i, :] = bk_mask[i, :] * GRAD_MAX * (ori_h / 2 - i) / (ori_h / 2)

    # bluring the background image with gradiental filter size
    blur_im = np.zeros(ori_im.shape, dtype=np.uint8)

    kernels = KERNEL_3[0]
    if not kernels % 2 == 1:
        kernels += 1
    for ker_sz in range(0, kernels, 2):
        top = max(int(ker_sz * ori_h / kernels) - int(kernels/2), 0)
        bottom = min(int((ker_sz + 2) * ori_h / kernels) + int(kernels/2), ori_h)

        mask = np.zeros((ori_h, ori_w), dtype=np.float)
        mask[top:bottom, :] = np.ones((bottom - top, ori_w), dtype=np.float)
        mask = cv2.GaussianBlur(mask*255, (kernels, kernels), 0)
        # mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]

        sub_blur_im = cv2.GaussianBlur(ori_im, (kernels - ker_sz, kernels - ker_sz), 0)[top:bottom, :]
        for ch in range(chs):
            blur_im[top:bottom, :, ch] = sub_blur_im[:, :, ch] * mask[top:bottom, :] / 255.0 +\
                                         blur_im[top:bottom, :, ch] * (1 - mask[top:bottom, :] / 255.0)

    # combine the blured background and cropped object image
    for ch in range(chs):
        rm_im[:, :, ch] = ori_im[:, :, ch] * (obj_mask[:, :] / 255.0)
        blur_im[:, :, ch] = blur_im[:, :, ch] * (1 - obj_mask[:, :] / 255.0) + \
                               rm_im[:, :, ch] * (obj_mask[:, :] / 255.0)
    rm_im[:, :, 3] = obj_mask  # alpha channel for .png file

    # adjust the brightness with lightness(from top to center) and darkeness(from center to bottom)
    blur_im = blur_im.astype(np.uint8)
    hsv_im = cv2.cvtColor(blur_im, cv2.COLOR_BGR2HSV)
    hsv_im[:, :, 2] = np.where((hsv_im[:, :, 2] + bk_mask[:, :] * GRAD_SCALE) > 255, 255,
                               (np.where((hsv_im[:, :, 2] + bk_mask[:, :] * GRAD_SCALE) < 0, 0,
                                         hsv_im[:, :, 2] + bk_mask[:, :] * GRAD_SCALE)))
    blur_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)

    rm_im = rm_im.astype(np.uint8)
    blur_im = blur_im.astype(np.uint8)

    return rect, color, rm_im, blur_im


def _text_proc(ori_im):

    ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
    text = image_to_string(Image.fromarray(ori_im), lang='eng')

    return text.replace('\n', ' ')


def _proc(ori_im, rm_thm):

    # detect info from image
    rect, color, obj_im, feather_im = _img_proc(ori_im, rm_thm)
    text = _text_proc(ori_im)

    return [rect, color, text], obj_im, feather_im


def scan_folder(ori_dir, out_dir, label):

    cnt = 1
    # Scan all files on origin folder
    files = [f for f in listdir(ori_dir) if isfile(join(ori_dir, f))]
    total = len(files)

    res_dics = []

    for f in files:

        fn, ext = os.path.splitext(f)

        if ext.lower() == '.jpg':

            rm_fn = fn + '-outputs.png'
            rm_thum_path = os.path.join(out_dir, rm_fn)  # file path of removed backgroud and cropped

            # rm_obj_fn = fn + '-car.png'
            # rm_obj_path = os.path.join(ori_dir, rm_obj_fn)  # path car without background with original size

            rm_out_fn = fn + '.png'
            rm_out_path = os.path.join(ori_dir, rm_out_fn)  # path for image with feathered background

            if os.path.isfile(rm_thum_path):
                # logger.log_print("    %d / %d, File Name: %s\n" % (cnt, total, f))
                sys.stdout.write("    %d / %d, File Name: %s\n" % (cnt, total, f))

                # print("%d / %d, File Name: %s\n" % (cnt, total, f))
                cnt += 1

                ori_path = os.path.join(ori_dir, f)
                ori_img = cv2.imread(ori_path)
                os.remove(ori_path)  # remove the temporary files

                rm_thumb_img = cv2.imread(rm_thum_path, -1)
                os.remove(rm_thum_path)  # remove the temporary files

                """ end proc for each image """
                [rect, color, text], obj_img, out_img = _proc(ori_img, rm_thumb_img)

                # cv2.imwrite(rm_obj_path, obj_img)  # write the object image which isolated from original background
                cv2.imwrite(rm_out_path, out_img)  # write the result image on original path

                # send()  # send image to S3 bucket
                path_key = "{}/{}".format(label, rm_out_fn)
                # s3_url = s3_manager.image_upload_to_s3(rm_out_path, path_key)
                s3_url = image_upload_to_s3(rm_out_path, path_key)

                dic = {RECT: rect, COLOR: color, TXT: text, CLASS: label, URL: s3_url}
                sys.stdout.write("    {}".format(dic))
                # logger.log_print("    {}".format(dic))

                res_dics.append(dic)

                # remove the result image from the ec2 instance
                os.remove(rm_out_path)
    # print("res_dics : ")
    # print(res_dics)
    return res_dics


def recog(origin_dir, output_dir, label):

    # get all images from origin_dir and output_dir
    origin_paths = glob.glob(os.path.join(origin_dir, "*.jpg"))
    output_paths = glob.glob(os.path.join(output_dir, "*.png"))

    if origin_dir is None or output_dir is None:
        raise Exception("input_dir or Output_dir not defined")

    if len(origin_paths) == 0:
        raise Exception("There is no image file to process")

    if len(output_paths) == 0:
        raise Exception("There is no image, which was ML processed")

    logger.log_print("    Scanning folder ...\n")
    res_dicts = scan_folder(origin_dir, output_dir, label)

    logger.log_print("    {} files Successfully end processed\n".format(len(res_dicts)))
    return res_dicts


