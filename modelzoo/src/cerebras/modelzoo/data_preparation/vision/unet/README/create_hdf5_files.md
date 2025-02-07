# Introduction
The Severstal dataset provides images of surface defects on steel sheets from high frequency cameras.
These images also come with labeled ground truth segmentation masks.
The goal of the dataset is develop algorithms to detect and localize these defects.
Although there are five classes corresponding to four types of defects and the absence of any defects, we use the dataset for binary classification (a specified defect type versus no defects).
Since the original dataset is suited for use with map-style dataset, we provide a script for resizing the input data and saving it in the HDF5 format which is more suited for iterable-style datasets.

## Input data
The document for the dataset is available on the [Kaggle competition page](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data).

### Data download
The dataset can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data).

## Input files format

The images are provided in JPG format with a CSV file `<split>.csv` with columns `ImageId`, `ClassId` and `EncodedPixels` corresponding to the image file name, the defect type and a list of pixels corresponding to the defect.

## Running the script
Before we run the script, we specify the arguments used in the script in the default config yaml:
1. Set image shape to desired shape in `train_input.image_shape` and `eval_input.image_shape` i.e. [H, W, 1] in config: /path_to_modelzoo/vision/pytorch/unet/configs/params_severstal_binary.yaml
2. Set the desired class to be considered in `train_input.class_id`
3. Set the desired `train` and `val` splits in `train_input.train_test_split`
4. Set the class to be considered as positive label in `train_input.class_id`


Once done, we run the script:
```
# For help:
python create_hdf5_files.py -h

# Run the script
python modelzoo/data_preparation/vision/unet/create_hdf5_files.py --params <path_to_modelzoo/vision/pytorch/unet/configs/params_severstal_binary.yaml> --output_dir <output dir> severstal_binary_classid_3_hdf --num_output_files <no. output files> --num_processes <no. processes for preprocessing>
```

## Data processors
The input dataset can be used with [SeverstalBinaryClassDataProcessor.py](../../../../data/vision/segmentation/SeverstalBinaryClassDataProcessor.py) which uses a map-style PyTorch Dataset while the output dataset is used with [Hdf5DataProcessor.py](../../../../data/vision/segmentation/Hdf5DataProcessor.py) which uses an iterable-style PyTorch.

## Output data
The input data format and structure is quite different compared to the input.
In particular, each HDF5 file contains multiple examples:
```

preprocessed_data-0_p0.h5
├── "example_0"
│   ├── "image": np.array
│   ├── "label": np.array
├── "example_2"
│   ├── "image": np.array
│   ├── "label": np.array
├── ...
```

## Output directory structure
```
├── data_params.json
├── eval_input
│   ├── preprocessed_data-0_p0.h5
│   ├── preprocessed_data-1_p0.h5
│   ├── ...
├── meta_eval_input.dat
├── meta_train_input.dat
├── train_input
│   ├── preprocessed_data-0_p0.h5
│   ├── preprocessed_data-1_p0.h5
│   ├── ...
```
