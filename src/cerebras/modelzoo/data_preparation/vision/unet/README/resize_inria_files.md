# Introduction
The Inria-Aerial dataset comprises of 300 high resolution aerial imagery of various geographical locations ranging from densely populated areas to alpine towns.
The images provided by the benchmark are of size 5000 x 5000.
Using image shapes which are powers of 2 yields the best performance on CS systems. Also, note that the shape of the input (height and width) passed to model should be such that the output feature maps from Encoder blocks are divisible by 2.
In this regard, we use a preprocessing script to create a resized dataset using resampling or center-crop transforms.

## Input data
Information of the dataset can be available in the [original paper](https://hal.inria.fr/hal-01468452/document).

### Data download
The dataset can be downloaded from the [Inria Aeria Image Labeling Dataset Webpage](https://project.inria.fr/aerialimagelabeling/) after registering as a user.

## Input files format
The image and label are presented as TIFF image file format.

## Running the script
The throughput run-time of the script vary depending on desired image size.
```
python modelzoo/data_preparation/vision/unet/resize_inria_files.py --input_dir <path to raw inria dataset folder> --output_dir <path to resized images folder> --width <resized width> --height <resized height> --transform <resize or center-crop>
```

## Data processors
This dataset can be used with [InriaAerialDataProcessor.py](../../../../data/vision/segmentation/InriaAerialDataProcessor.py) which uses a map-style PyTorch Dataset.
We set aside 15 images (`train/images/austin{34,19,10}.tif`, `train/images/chicago{6,22,18}.tif`, `train/images/kitsap{18,4,26}.tif`, `train/images/tyrol-w{19,26,22}.tif`, `train/images/vienna{1,23,18}.tif`) to be used as validation to measure model performance using mIOU metrics.

## Output data
We keep the same train/val/test splits as the original datasets, changing only the image and label dimensions.

## Output directory structure
The output directory structure largely follows the original format.
```
.
├── test
│   ├── images
│   │   ├── bellingham10.tif
│   │   ├── bellingham11.tif
│   │   ├── ...
├── train
│   ├── gt
│   │   ├── austin11.tif
│   │   ├── austin12.tif
│   │   ├── ...
│   ├── images
│   │   ├── austin11.tif
│   │   ├── austin12.tif
│   │   ├── ...
├── val
│   ├── gt
│   │   ├── austin10.tif
│   │   ├── austin19.tif
│   │   ├── ...
│   ├── image
│   │   ├── austin10.tif
│   │   ├── austin19.tif
│   │   ├── ...
```