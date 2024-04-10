# Introduction
The Stanford Knee MRI with Multi-Task Evaluation (SKM-TEA) is a dataset containing quantitative knee MRI scans of 155 patients with a total of 25,000 slices.
This dataset can be used with the UNet3D model provided in the Cerebras Model Zoo for 3D
segmentation training.
While the majority of the MRI volumes are of size 128 x 512 x 512, a few of them differ in the depth dimension.
To fix this, we provide a preprocessing script to resize the scans to an arbitrary volume size using nearest interpolation.

# Environment setup
A minimal Python environment is needed for the script.
```
python -m venv venv_skm_tea
source venv_skm_tea/bin/activate
pip install h5py pyyaml tqdm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## Input data
The documentation for the raw data is available on the [official repository](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md).

### Data download
Data can be downloaded from the [Stanford Artifical Intelligence and Medical Imaging](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7). Note that the full dataset is around 1.6 TB but the data needed for the segmentation task is around 17 GB.
In particular, we only need the extracted data from `image_files.tar.gz`.

## Input files format
A high level overview of the data is that each MRI scan corresponds to a HDF5 file `v1-release/image_files/MTR_<patient_id>.h5` which contains that contains the echo data (keys `echo1`, `echo2`) and the segmentation mask (key `seg`) as well as some statistical data (key `stats`) containing basic statistics (min, max, mean, std) of the MRI scan.
Information for the dataset split is found in `v1-release/annotations/v1.0.0/<split>.json`, which is similiar to the COCO annotation format.

## Running the script
The throughput run-time of the script vary depending on desired volume size.
```
python modelzoo/data_preparation/vision/unet/resize_skm_tea_files.py --input_dir <path to raw dataset> --output_dir <output dir path> --width <resized width> --height <resized height> --depth <resized depth>
```

## Data processors
The output dataset is to be used with [SkmDataProcessor](../../../../data/vision/segmentation/SkmDataProcessor.py) which uses a iterable-style dataset.

## Output data
We follow the original data format for the resized volumes, namely that each HDF5 file corresponds to a MRI scan.
However, we opted to store only the echo data and segmentation mask and omit the statistical metrics since they are not used in the dataloader preprocessing.
For the dataset splits, the associated JSON files are updated with the new volume sizes.

Similarly, the volume contains only the necessary keys
```
MTR_001.h5
├── "echo1": np.array # Echo 1 of the qDESS scan
├── "echo2": np.array # Echo 2 of the qDESS scan
├── "seg": np.array # One-hot encoded segmentations
```

## Output directory structure
The output directory structure largely follows the original format except it contains only the folders and files needed for segmentation.
```
.
v1_release
├── annotations
│   ├── v1.0.0
│   │   ├── test.json
│   │   ├── train.json
│   │   ├── val.json
├── image_files
│   ├── MTR_001.h5
│   ├── MTR_005.h5
│   ├── ...
```
