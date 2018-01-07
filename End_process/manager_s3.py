import os
import sys
import boto
from boto.s3.key import Key

"""
bucket: lotlinx-ml-images
access key: AKIAJKLDTATHUX222RUA
secret key: 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
"""
BUCKET_NAME = "lotlinx-ml-images"
AWS_ACCESS_KEY_ID = "AKIAJKLDTATHUX222RUA"
AWS_SECRET_KEY = "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"

# get the service client
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_KEY)
if conn is None:
    sys.stdout.write("Failed to connect the S3")
bucket = conn.get_bucket(BUCKET_NAME)

if bucket is None:
    sys.stdout.write("No such Bucket" + BUCKET_NAME)


def upload_to_s3(fpath):
    buck_k = Key(bucket)

    buck_k.key = "classname/" + fname  # file key on s3 bucket
    buck_k.set_contents_from_filename(fpath)

    buck_k.set_acl('pulic_read')
    # os.remove(fpath)


def image_upload_to_s3(fpath, path_key):
    buck_k = Key(bucket)
    buck_k.key = path_key
    buck_k.set_contents_from_filename(fpath)

    buck_k.set_acl('public-read')
    img_url = buck_k.generate_url(0, query_auth=False, force_http=True)
    return img_url


def download_all_from_s3():
    bucket_list = bucket.list()
    for l in bucket_list:
        keyString = str(l.key)
        if not os.path.exists(keyString):
            l.get_contents_to_filename(keyString)


def download_from_s3(path_key, fpath):

    key = bucket.get_key(path_key)
    if not os.path.exists(fpath):
        key.get_contents_to_filename(fpath)

