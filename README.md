# outline

Based on [Image to Image Translate(CGAN)](https://bitbucket.org/Din_Osh/outline.git) by Isola et al.

[Article about this implemention](https://affinelayer.com/pix2pix/)

Tensorflow implementation of pix2pix.  Learns a mapping from input images to output images, like these examples from the original paper:

The processing speed on a GPU with cuDNN was equivalent to the Torch implementation in testing.

# Setup

### Prerequisites
- Tensorflow 1.0.0
- opencv 3.2.0
- python 2.7 or 3.5 anaconda
- Pytesseract and tesseract (important: move download English language data for the OCR engine:)


### Recommended
- Ubuntu 14.04 AWS EC2 Instance
- Linux with Tensorflow GPU edition + cuDNN
- CUDA 8.0
- cuDNN-8.0 5.1

### Project Details

- CGAN_model
    * tools
        * process.py
        * split.py
    * pix2pix.py

- Inception_model
    * train_data.py
    * softmax.py
        * model/

- End_processing
    * detect_process.py

- Pre_processing
    * align_process.py
- data
    * samples/
        * front/
        * front_3_quarter/
        * rear/
        * rear_3_quarter/
        * side/
        * interior/
        * tire/


# CGAN_model

## Training

```sh
# train the model (this may take over 10 hours depending on GPU, on CPU you will be waiting for a bit)
python CGAN_model/pix2pix.py \
  --mode ... \
  --output_dir CGAN_model/model/... \
  --max_epochs 500 \
  --input_dir data/samples... \
  --which_direction AtoB \
  --checkpoint ...

# init training
python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/samples/front_3_quarter/combined/train \
  --which_direction AtoB \

# from checkpoints
python CGAN_model/pix2pix.py \
  --mode train \
  --output_dir CGAN_model/model/front_3_quarter \
  --max_epochs 500 \
  --input_dir data/samples/front_3_quarter/combined/train \
  --which_direction AtoB \
  --checkpoint CGAN_model/model/front_3_quarter
```

In this mode, image A is the black and white image (lightness only), and image B contains the color channels of that image (no lightness information).


## Testing

Testing is done with `--mode test`.  You should specify the checkpoint to use with `--checkpoint`, this should point to the `output_dir` that you created previously with `--mode train`:

```sh
python CGAN_model/pix2pix.py \
  --mode test \
  --output_dir ...\
  --input_dir  ...\
  --checkpoint ...
```

Test the model with checkpoints
```sh
python CGAN_model/pix2pix.py \
  --mode test \
  --output_dir input_dir data/samples/front_3_quarter/combined/test_result\
  --input_dir input_dir data/samples/front_3_quarter/combined/val\
  --checkpoint CGAN_model/model/front_3_quarter
```

The testing mode will load some of the configuration options from the checkpoint provided so you do not need to specify `which_direction` for instance.

# Inception_model

## Prepare the training data
```sh
python Inception_model/train_data.py \
  --input_dir data/samples \
  --output_dir Inception_model/model \
  --mode train
```

## Training
```
# init training

python Inception_model/softmax.py \
  --input_dir Inception_model/model \
  --output_dir Inception_model/model \
  --restore no \
  --rate 0.0001 \
  --epochs 200000 \
  --strip 50

or from checkpoint

python Inception_model/softmax.py \
  --input_dir Inception_model/model \
  --output_dir Inception_model/model \
  --restore yes \
  --strip 500
```

## Testing
```sh
```


# Datasets and Trained Models

The data format used by this program is the same as the original image to image translate format, which consists of images of input and desired output side by side like:

Some datasets have been made available by the authors of the pix2pix paper.  To download those datasets, use the included script `tools/download-dataset.py`.  There are also links to pre-trained models alongside each dataset, note that these pre-trained models require the Tensorflow 0.12.1 version of pix2pix.py since they have not been regenerated with the 1.0.0 release:

| dataset | example |
| --- | --- |

## Pre_Process

### Align the training data
Resizing and register with the image pairs target(.png) and source(.jpg) 
```sh
# align and resize the source images
python Pre_process/align_process.py \
  --input_dir data/samples/front_3_quarter \
  --mode align \
  --size 256
```

### Combine the A folder images and B folder images
If you have two directories `a` and `b`, with corresponding images (same name, same dimensions, different data) you can combine them with `process.py`:

```sh
python tools/process.py \
  --input_dir a \
  --b_dir b \
  --operation combine \
  --output_dir c
```

Creating image pairs from existing images for testing
This puts the images in a side-by-side combined image that `pix2pix.py` expects.

```sh
python CGAN_model/tools/process.py \
  --input_dir data/samples/front_3_quarter/A \
  --b_dir data/samples/front_3_quarter/B \
  --operation combine \
  --output_dir data/samples/front_3_quarter/combined
```
This puts the images in a side-by-side combined image that `pix2pix.py` expects.

### Split into train/val set
```sh
python CGAN_model/tools/split.py \
  --dir data/samples/front_3_quarter/combined
```
The folder `./combined` will now have `train` and `val` subfolders that you can use for training and testing.

## End_process

### End process
```sh
python End_process/detect_process.py \
  --origin_dir data/temp \
  --output_dir data/temp/output/images \
  --size 256
```

# Implementation

### Pre_process
### Classification_process with Inception_model
### Main_Process with CGAN_model
### End_process


## Code Validation
Validation of the code was performed on a Linux machine with a Ubuntu 14.04, Nvidia GTX 980 Ti GPU and an AWS P2 instance with a K80 GPU.


## Call the API with curl

curl -X POST -H "Content-Type: multipart/form-data" -F "file=@test.json" http://34.227.32.180:5000/submit

curl -v -X POST -H "Content-Type: multipart/form-data" -F "file=@test.json" http://34.227.32.180:5000/submit

curl -o resutl.json -v -X POST -H "Content-Type: multipart/form-data" -F "file=@test.json" http://34.227.32.180:5000/submit

## Acknowledgments
This is a port of [pix2pix](https://github.com/phillipi/pix2pix) from Torch to Tensorflow.  It also contains colorspace conversion code ported from Torch.  Thanks to the Tensorflow team for making such a quality library!  And special thanks to Phillip Isola for answering my questions about the pix2pix code.
