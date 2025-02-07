# Dataset Preparation
- [Dataset Preparation](#dataset-preparation)
  - [ImageNet (ILSVRC2012)](#imagenet-ilsvrc2012)
    - [What to do if you are missing one or more files](#what-to-do-if-you-are-missing-one-or-more-files)

## ImageNet (ILSVRC2012)
We use torchvision.datasets.ImageNet to create our dataset. We assume that the data has already been extracted and pre-processed. Specifically, the dataset root directory must already have the following structure:
```
root_directory
├── meta.bin
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── ...
│   ├── n01443537
│   │   ├── n01443537_10007.JPEG
│   │   ├── n01443537_10014.JPEG
│   │   ├── n01443537_10025.JPEG
│   │   ├── ...
│   ├── ...
│   └── ...
│   val/
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000946.JPEG
│   │   ├── ILSVRC2012_val_00001684.JPEG
│   │   └── ...
│   ├── n01443537
│   │   ├── ILSVRC2012_val_00001269.JPEG
│   │   ├── ILSVRC2012_val_00002327.JPEG
│   │   ├── ILSVRC2012_val_00003510.JPEG
│   │   └── ...
│   ├── ...
│   └── ...
```

### What to do if you are missing one or more files
If you don't have `meta.bin` in your directory, you should download ILSVRC2012_devkit_t12.tar.gz with:
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

If you don't have `train/` and/or `val/` directories in this format, you can find the original dataset on the ImageNet official website (https://image-net.org/download.php). You may need to register and request download permission. After download, you should have `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`. 

Once you have all three tar files, you would need to extract and preprocess the archives into the appropriate directory structure. The simplest way is to initialize torchvision.datasets.ImageNet once.

```
import torchvision
root_dir = <path_to_ILSVRC2012_img_{split}.tar>
torchvision.datasets.ImageNet(root=root_dir, split="train")
torchvision.datasets.ImageNet(root=root_dir, split="val)
```
