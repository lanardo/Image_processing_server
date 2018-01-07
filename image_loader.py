import sys
import grequests
import numpy as np
import json
# from urllib.parse import urlparse  # python 3x
from urlparse import urlparse  # python 2x
import os
import cv2

import logger

# Feilds
DEARLERID = "\"dealerid\""
VEHICLES = "vehicles"
IMAGES = "images"
URL = "url"

save_dir = None
fnames = []


def parse_json(json_fname):
    # load json file
    with open(json_fname, 'r') as f:
        data = f.read()
        # data = data.replace(" ", "")
        str_data = data.replace("\n", "")

    # for the java request
    while hex(ord(str_data[-1])) != hex(0x7d):
        str_data = str_data[:-1]

    # logger.log_print(str_data + "\n")
    # with open("str_data", 'w') as f:
    #     f.write(str_data)
    # print("\n")

    dealers = []
    # parsing the json file with string
    # this is because of json format
    pos = 0
    while pos < len(str_data):
        start_pos = str_data.find(DEARLERID, pos)
        pos = str_data.find(DEARLERID, start_pos + 1)
        if pos == -1:
            pos = len(str_data)

        if str_data[pos - 2] != ",":
            sub_str = "{" + str_data[start_pos:pos - 1] + "}"
        else:
            sub_str = "{" + str_data[start_pos:pos - 2] + "}"

        # with open("temp{} - {}".format(start_pos, pos), 'w') as f:
        #     f.write(sub_str)
        #
        # print("sub_str")
        # logger.log_print("{}\n".format(sub_str))

        dealer = json.loads(sub_str)
        dealers.append(dealer)

        # print("dealer")
        # logger.log_print("{}\n".format(dealer))

    return dealers


def get_urls(dealer_list):

    url_list = []
    for dealer in dealer_list:
        for vehicle in dealer[VEHICLES]:
            for image in vehicle[IMAGES]:
                url_list.append(image[URL])

    return url_list


def load_images_from_urls(urls, out_dir, return_responses=False):

    requests = (grequests.get(url) for url in urls)
    responses = grequests.map(requests)
    if return_responses:
        return responses

    img_fns = []
    failed_urls = []
    for i in range(len(responses)):
        response = responses[i]
        if response is not None:
            if 199 < response.status_code < 400:
                im = np.asarray(bytearray(response.content), dtype="uint8")
                img = cv2.imdecode(im, cv2.IMREAD_COLOR)
                if img is not None:

                    a = urlparse(urls[i])
                    img_fn = os.path.basename(a.path)
                    img_fns.append(img_fn)

                    img_path = os.path.join(out_dir, img_fn)
                    cv2.imwrite(img_path, img)
                    # logger.log_print("    {}/{} success file : {}\n".format(i+1, len(responses), img_fn))
                    sys.stdout.write("    {}/{} success file : {}\n".format(i + 1, len(responses), img_fn))

            else:
                # a = urlparse(urls[i])
                # img_fn = os.path.basename(a.path)
                sys.stdout.write(
                    "    {}/{} failed with error {} file : {}\n".format(i + 1, len(responses), response.status_code, urls[i]))
                failed_urls.append(urls[i])
        else:
            sys.stdout.write(
                "    {}/{} failed with None response file : {}\n".format(i + 1, len(responses), urls[i]))
            failed_urls.append(urls[i])

        sys.stdout.flush()

    return img_fns, failed_urls


def load(json_fn, out_dir):

    # parse the json file
    dealers = parse_json(os.path.join(out_dir, json_fn))
    os.remove(os.path.join(out_dir, json_fn))

    # get urls from dealers of json data
    urls = get_urls(dealers)

    # download the images from its urls
    img_fns, fails = load_images_from_urls(urls, out_dir)

    logger.log_print("    succeed:{}| failed:{} | total:{}\n".format(len(img_fns), len(fails), len(urls)))
    retry = 1
    new_fails = []
    while retry > 0:
        retry -= 1
        if len(fails) != 0:
            new_fns, new_fails = load_images_from_urls(fails, out_dir)
            img_fns.extend(new_fns)
        else:
            break
        fails = new_fails
        logger.log_print("    succeed:{}| failed:{}\n".format(len(new_fns), len(new_fails)))

    logger.log_print("    failed files{}\n".format(len(new_fails)))
    return dealers, urls, img_fns

# if __name__ == '__main__':
#     load("ml-input-multi.json", "./data/json_files")
