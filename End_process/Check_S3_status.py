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


def image_upload_to_s3(fpath, path_key):
    buck_k = Key(bucket)
    buck_k.key = path_key
    buck_k.set_contents_from_filename(fpath)

    buck_k.set_acl('public-read')
    img_url = buck_k.generate_url(0, query_auth=False, force_http=True)
    print(img_url)


def get_list_of_bucket():
    for key in bucket.list():
        print(key.name)


def empty_bucket():
    for key in bucket.list():
        bucket.delete_key(key)


def del_object_from_bucket(folder_path, file_name):
    k = Key(bucket)
    k.key = folder_path + file_name
    bucket.delete_key(k)


if __name__ == '__main__':

    # image_upload_to_s3('test.bmp', 'test.bmp')
    # print("all lists")
    empty_bucket()

    print("all lists")
    get_list_of_bucket()
