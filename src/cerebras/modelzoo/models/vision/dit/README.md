# Diffusion Transformer

- [Diffusion Transformer](#diffusion-transformer)
  - [Model overview](#model-overview)
  - [Structure of the code](#structure-of-the-code)
  - [Sequence of the steps to perform](#sequence-of-the-steps-to-perform)
    - [Step 1: ImageNet dataset download and preparation](#step-1-imagenet-dataset-download-and-preparation)
    - [Step 2: Checkpoint Conversion of Pre-trained VAE](#step-2-checkpoint-conversion-of-pre-trained-vae)
    - [Step 3: Preprocessing and saving Latent tensors from images and VAE Encoder on GPU](#step-3-preprocessing-and-saving-latent-tensors-from-images-and-vae-encoder-on-gpu)
        - [a. Create ImageNet Latent Tensors from VAE for `train` split of dataset](#a-create-imagenet-latent-tensors-from-vae-for-train-split-of-dataset)
        - [b. Create ImageNet Latent Tensors from VAE for `val` split of dataset](#b-create-imagenet-latent-tensors-from-vae-for-val-split-of-dataset)
        - [c. Create ImageNet Latent Tensors with horizontal flip from VAE for `train` split of dataset](#c-create-imagenet-latent-tensors-with-horizontal-flip-from-vae-for-train-split-of-dataset)
        - [d. Create ImageNet Latent Tensors with horizontal flip from VAE for `val` split of dataset](#d-create-imagenet-latent-tensors-with-horizontal-flip-from-vae-for-val-split-of-dataset)
    - [Step 4: Training the model on CS system or GPU using `run.py`](#step-4-training-the-model-on-cs-system-or-gpu-using-runpy)
      - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
      - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
    - [Step 5: Generating 50K samples from trained checkpoint on GPUs from FID score computation](#step-5-generating-50k-samples-from-trained-checkpoint-on-gpus-from-fid-score-computation)
    - [Step 6: Using OpenAI FID evaluation repository to compute FID score](#step-6-using-openai-fid-evaluation-repository-to-compute-fid-score)
      - [a. Set up a conda environment to use OpenAI evaluation script](#a-set-up-a-conda-environment-to-use-openai-evaluation-script)
      - [b. Clone OpenAI guided-diffusion GitHub repository](#b-clone-openai-guided-diffusion-github-repository)
      - [c. Download the `npz` files corresponding to reference batch of ImageNet](#c-download-the-npz-files-corresponding-to-reference-batch-of-imagenet)
      - [d. Make changes to `evaluator.py`](#d-make-changes-to-evaluatorpy)
      - [e. Launch FID eval script with the following command](#e-launch-fid-eval-script-with-the-following-command)
  - [Configuration files included for this model](#configuration-files-included-for-this-model)
  - [DataLoader Features Dictionary](#dataloader-features-dictionary)
  - [Implementation notes](#implementation-notes)
  - [Citations](#citations)


## Model overview

This directory contains implementation for Diffusion Transformer (DiT). Diffusion Transformer[[1](https://arxiv.org/pdf/2212.09748.pdf)], as the name suggests, belongs to the class of diffusion models. However, the key difference is that it replaces the UNet architecture backbone typically used in previous diffusion models with a Transformer backbone and some modifications. This model beats the previous diffusion models in FID-50K[[7](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)] eval metric.

<p align="center">
    <img src="./images/dit_adalnzero.png">
</p>
<p align="center">
    Figure 1: DiT Model with AdaLN-Zero. The dimensions are for an image of size 256 x 256 x 3
</p>


A DiT model consists of `N` layers of DiT blocks. We support the following variants of DiT Block. More details can be found in the Section 3.2 of the paper [[1](https://arxiv.org/pdf/2212.09748.pdf)] and [Step 4: Training the model on CS system or GPU using `run.py`](#step-4-training-the-model-on-cs-system-or-gpu-using-runpy)

In addition, we also support patch sizes of 2 (default), 4 and 8. The `Patchify` block in Figure 1 takes noised latent tensor as input from dataloader and converts into patches of 
size `patch_size`. The lower the patch size, the larger the number of patches (i.e maximum sequence length (MSL)) and hence larger the number of FLOPS. 

In order to change the patch size used, for example to `4 x 4`, set `model.patch_size: [4, 4]` in yaml config. Details on all the configs provided can be found [here](#configuration-files-included-for-this-model).

During training, an image from the dataset is taken and passed through a frozen VAE Encoder (Variational Auto Encoder)[[8](https://arxiv.org/pdf/1312.6114.pdf)] to convert the image into a lower dimensional latent. Then, random gaussian noise is added to the latent tensor (Algorithm 1 of [[2](https://hojonathanho.github.io/diffusion/assets/denoising_diffusion20.pdf)) and passed as input to the DiT .Since the VAE Encoder[[8](https://arxiv.org/pdf/1312.6114.pdf)] is frozen and not updated during the training process, we prefetch the latents for all the images in the dataset using the script [create_imagenet_latents.py](./input/scripts/create_imagenet_latents.py). This helps save computation and memory during the training process. Refer to [Section here](#step-3-preprocessing-and-saving-latent-tensors-from-images-and-vae-encoder-on-gpu).

## Structure of the code

-   `configs/`: YAML configuration files.
-   `data.py`: The entry point to the data input pipeline code. Defines `train_input_dataloader` ( and `eval_input_dataloader`) which initalizes the data processor mentioned in config yaml `train_input.data_processor`( and `eval_input.data_processor`)
-   `modeling_dit.py`: Defines the core model `DiT`.
-   `model.py`: The entry point to the model. Defines `DiTModel`.
-   `run.py`: Training script. Performs training and validation.
-   `utils.py`: Miscellaneous scripts to parse the `params` dictionary from the YAML files.
-   (data/vision/diffusion/)(../../../data/vision/diffusion): Folder containing Dataloader and preprocessing scripts.
-   `samplers/`: Folder containing samplers used in Diffusion models to sample images from checkpoints.
-   `layers/vae/`: Defines VAE(Variational Auto Encoder) model layers.
-   `layers/*`: Defines building block layers of DiT Model.
-   (tools/checkpoint_converters/internal/vae_hf_cs.py)(../../../tools/checkpoint_converters/internal/vae_hf_cs.py): Converts [pretrained VAE checkpoint from StabilityAI in HuggingFace](https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin") to CS namespace format. 
-   `display_images.py`: Utility script to display images in a folder in a grid format to look at all images at once.
-   `pipeline.py`: Defines a DiffusionPipeline object that takes in a random gaussian input and performs sampling.
-   `sample_generator.py`: Defines a Abstract Base Class `SampleGenerator` to define sample generators for diffusion models.
-   `sample_generator_dit.py`: Defines a `DiTSampleGenerator` that inherits from `SampleGenerator` class and is used to generate required number of samples from DiT Model using a given trained checkpoint.



## Sequence of the steps to perform

The high-level steps for training a model are relatively simple, involving data-processing and then model training and evaluation

* Step 1: ImageNet dataset download and preparation
* Step 2: Checkpoint Conversion of Pre-trained VAE.
* Step 3: Preprocessing and saving Latent tensors from images and VAE Encoder on GPU
* Step 4: Training the model on CS system or GPU using `run.py`
* Step 5: Generating 50K samples from trained checkpoint on GPUs
* Step 6: Using OpenAI FID evaluation repository to compute FID score.

The steps are elaborated below:

### Step 1: ImageNet dataset download and preparation
Inorder to download the ImageNet dataset, register on the [ImageNet website](http://image-net.org/)[[4](https://www.image-net.org/challenges/LSVRC/2012/)]. The dataset can only be downloaded after the ImageNet website confirms the registration and sends a confirmation email. Please follow up with [ImageNet support](support@image-net.org) if a confirmation email is not received within a couple of days.

Download the tar files `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz` for the ImageNet dataset.

Once we have all three tar files, we would need to extract and preprocess the archives into the appropriate directory structure as described below. 

```
root_directory (imagenet1k_ilsvrc2012)
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
Inorder to arrange the ImageNet dataset in the above format, Pytorch repository provides an easy to use script that can be found [here: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh). Download this script and invoke it as follows to preprocess the ImageNet dataset.
```
source extract_ILSVRC.sh
```

We also need a `meta.bin` file. The simplest way is to create it is to initialize `torchvision.datasets.ImageNet` once.

```
import torchvision
root_dir = <path_to_root_dir_imagenet1k_ilsvrc2012_above>
torchvision.datasets.ImageNet(root=root_dir, split="train")
torchvision.datasets.ImageNet(root=root_dir, split="val)
```

Once the ImageNet dataset and folder are in the expected format, proceed to Step 2

### Step 2: Checkpoint Conversion of Pre-trained VAE

The next step is to convert the pretrained checkpoint provided by StabilityAI and hosted on HuggingFace to CS namespace format. This can be done using the script [vae_hf_cs.py](../../../tools/checkpoint_converters/internal/vae_hf_cs.py). The script downloads the [pretrained VAE checkpoint from StabilityAI in HuggingFace](https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin") and converts to CS namespace based on model layers defined in [dit/layers/vae](./layers/vae/). For this script, we only care about the params defined under `model.vae_params` and no changes are needed except for setting the `model.vae_params.latent_size` correctly.


```
$ python modelzoo/tools/checkpoint_converter/internal/vae_hf_cs.py -h
usage: vae_hf_cs.py [-h] [--src_ckpt_path SRC_CKPT_PATH] [--dest_ckpt_path DEST_CKPT_PATH]
                    [--params_path PARAMS_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --src_ckpt_path SRC_CKPT_PATH
                        Path to HF Pretrained VAE checkpoint .bin file. If not provided, file is automatically
                        downloaded from https://huggingface.co/stabilityai/sd-vae-ft-
                        mse/resolve/main/diffusion_pytorch_model.bin (default: None)
  --dest_ckpt_path DEST_CKPT_PATH
                        Path to converted modelzoo compatible checkpoint (default:
                        modelzoo/models/vision/dit/checkpoint_converter/mz_stabilityai-sd-vae-ft-mse_ckpt.bin)
  --params_path PARAMS_PATH
                        Path to VAE model params yaml (default: modelzoo/models/vision/dit/configs/params_dit_small_patchsize_2x2.yaml)
```

Command to run:
```
python modelzoo/tools/checkpoint_converter/internal/vae_hf_cs.py --dest_ckpt_path=/path/to/save/converted/checkpoint
```

### Step 3: Preprocessing and saving Latent tensors from images and VAE Encoder on GPU

For training the DiT model, we prefetch the latent tensor outputs from a pretrained VAE Encoder using the script [`create_imagenet_latents.py`](../../../data_preparation/vision/dit/create_imagenet_latents.py)

```
$ python modelzoo/data_preparation/vision/dit/create_imagenet_latents.py -h
usage: create_imagenet_latents.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--params_path PARAMS_PATH] [--horizontal_flip]
                                  --image_height IMAGE_HEIGHT --image_width IMAGE_WIDTH --src_dir SRC_DIR --dest_dir DEST_DIR [--resume]
                                  [--resume_ckpt RESUME_CKPT] [--log_steps LOG_STEPS]
                                  [--batch_size_per_gpu BATCH_SIZE_PER_GPU] [--num_workers NUM_WORKERS]
                                  [--dataset_split {train,val}]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Path to VAE model checkpoint (default: None)
  --params_path PARAMS_PATH
                        Path to VAE model params yaml (default:modelzoo/models/vision/dit/configs/params_dit_small_patchsize_2x2.yaml)
  --horizontal_flip     If passed, flip image horizonatally (default: False)
  --image_height IMAGE_HEIGHT
                        Height of the resized image
  --image_width IMAGE_WIDTH
                        Width of the resized image
  --src_dir SRC_DIR     source data location (default: None)
  --dest_dir DEST_DIR   Latent data location (default: None)
  --resume              If specified, resumes previous generation process.The dest_dir should point to previous generation
                        and have log_checkpoint saved. (default: False)
  --resume_ckpt RESUME_CKPT
                        log ckpt to resume data generation fromIf None, picks latest from log dir (default: None)
  --log_steps LOG_STEPS
                        Generation process ckpt and logging frequency (default: 1000)
  --batch_size_per_gpu BATCH_SIZE_PER_GPU
                        batch size of input to be passed to VAE model for encoding (default: 64)
  --num_workers NUM_WORKERS
                        Number of pytorch dataloader workers (default: 4)
  --dataset_split {train,val}
                        Number of pytorch dataloader workers (default: train)

```

Inorder to preprocess the ImageNet dataset and create latent tensors, **using a GPU or multiple GPUs is required**.

Sample command is as follows for a single node with 4 GPUs. In the following command, we are using an image of size 256 x 256 and saving the latents to a folder specified by `--dest_dir`. The command also specifies to log every 10 steps and use a batch size of 16 i.e 16 images are batched together and passed to the VAE Encoder on each GPU.

##### a. Create ImageNet Latent Tensors from VAE for `train` split of dataset
```
torchrun --nnodes 1 --nproc_per_node 4 modelzoo/data_preparation/vision/dit/create_imagenet_latents.py --image_height=256 --image_width=256 --src_dir=/path/to/imagenet1k_ilsvrc2012 --dest_dir=/path_to_dest_dir --log_steps=10 --dataset_split=train --batch_size_per_gpu=16 --checkpoint_path=/path/to/converted/vae_checkpoint/in_Step2
``` 
##### b. Create ImageNet Latent Tensors from VAE for `val` split of dataset
```
torchrun --nnodes 1 --nproc_per_node 4 modelzoo/data_preparation/vision/dit/create_imagenet_latents.py --image_height=256 --image_width=256 --src_dir=/path/to/imagenet1k_ilsvrc2012 --dest_dir=/path_to_dest_dir --log_steps=10 --dataset_split=val --batch_size_per_gpu=16 --checkpoint_path=/path/to/converted/vae_checkpoint/in_Step2
```

The output folder shown below for reference and will have the same format as shown in Step 1:
```
/path_to_dest_dir 
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.npz
│   │   ├── n01440764_10027.npz
│   │   ├── n01440764_10029.npz
│   │   ├── ...
│   ├── n01443537
│   │   ├── n01443537_10007.npz
│   │   ├── n01443537_10014.npz
│   │   ├── n01443537_10025.npz
│   │   ├── ...
│   ├── ...
│   └── ...
│   val/
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000946.npz
│   │   ├── ILSVRC2012_val_00001684.npz
│   │   └── ...
│   ├── n01443537
│   │   ├── ILSVRC2012_val_00001269.npz
│   │   ├── ILSVRC2012_val_00002327.npz
│   │   ├── ILSVRC2012_val_00003510.npz
│   │   └── ...
│   ├── ...
│   └── ...
```

DiT models use horizontal flip of images as augmentation. The script also supports saving latent tensors from horizontally flipped images by passing the flag `--horizontal_flip`
##### c. Create ImageNet Latent Tensors with horizontal flip from VAE for `train` split of dataset
```
torchrun --nnodes 1 --nproc_per_node 4 modelzoo/models/vision/dit/input/scripts/create_imagenet_latents.py --image_height=256 --image_width=256 --src_dir=/path/to/imagenet1k_ilsvrc2012 --dest_dir=/path_to_hflipped_dest_dir --log_steps=10 --dataset_split=train --batch_size_per_gpu=16 --checkpoint_path=/path/to/converted/vae_checkpoint/in_Step2 --horizontal_flip
``` 
##### d. Create ImageNet Latent Tensors with horizontal flip from VAE for `val` split of dataset

```
torchrun --nnodes 1 --nproc_per_node 4 modelzoo/models/vision/dit/input/scripts/create_imagenet_latents.py --image_height=256 --image_width=256 --src_dir=/path/to/imagenet1k_ilsvrc2012 --dest_dir=/path_to_hflipped_dest_dir --log_steps=10 --dataset_split=val --batch_size_per_gpu=16 --checkpoint_path=/path/to/converted/vae_checkpoint/in_Step2 --horizontal_flip
```

### Step 4: Training the model on CS system or GPU using `run.py`

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a training run, make sure that in the YAML config file being used has the following set correctly:

-   The `train_input.data_dir` parameter points to the correct dataset
-   The `train_input.image_size` parameter corresponds to the image_size of the dataset.
-   The `model.vae.latent_size` parameter corresponds size of latent tensors. 
    -  Set to `[32, 32]` for image size of `256 x 256`
    -  Set to `[64, 64]` for image size of `512 x 512`
    -  In general, set to `[floor(H / 8), floor(W / 8)]` for an image size of `H x W`
-   The `model.patch_size` parameter to use different patch sizes

**To use with image size `512 x 512`, please make the following changes:**
1. `train_input.image_size`: [512, 512]
2. `model.vae.latent_size`: [64, 64]
3. `train_input.transforms`(if any): change `size` params under various transforms to [512, 512] 

**YAML config files**: Details on the configs for this model can be found in [Configs included for this model](#configs-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
-   `/path/to/model_dir` is a path to the directory where we would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

#### To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

#### To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../../../PYTHON-SETUP.md), and simply run:

```
python run.py {CPU,GPU} --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

### Step 5: Generating 50K samples from trained checkpoint on GPUs from FID score computation

Diffusion models report Fréchet inception distance (FID)[[7]](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf) metric on 50K samples generated from the trained checkpoint. In order to generate samples, we use a DDPM Sampler [[2](https://hojonathanho.github.io/diffusion/assets/denoising_diffusion20.pdf)] and without guidance (`model.reverse_process.guidance_scale=1.0`). Using a `model.reverse_process.guidance_scale >1.0` enables classifier free guidance which trades off diversity for sample quality.

The sample generation settings can be found in `model.reverse_params` in config yaml. We support two samplers cuurently, the [DDPM Sampler](./samplers/DDPMSampler.py)[[2](https://hojonathanho.github.io/diffusion/assets/denoising_diffusion20.pdf)] and [DDIM Sampler](./samplers/DDIMSampler.py)[[3](https://arxiv.org/pdf/2010.02502.pdf)]. All arguments in the `__init__` of the samplers can be set in the yaml config `model.reverse_params.sampler` section. 


To generate samples from a trained DiT checkpoint, we use GPUs and [sample_generator_dit.py](./sample_generator_dit.py).
Sample command to run on a single node with 4 GPUs to generate 50000 samples using trained DiT-XL/2. Each GPU uses a batch size of 64 and generates 64 samples at once. `--num_fid_samples` controls the number of samples to generate. This script cares about the section `model.reverse_params` in config yaml. Make sure that the settings are appropriate.

```
torchrun --nnodes 1 --nproc_per_node 4 modelzoo/models/vision/dit/sample_generator_dit.py --model_ckpt_path /path/to/trained/dit_checkpoint --vae_ckpt_path /path/to/converted/vae_checkpoint/in_Step2 --params modelzoo/models/vision/dit/configs/params_dit_xlarge_patchsize_2x2.yaml --sample_dir=/path/to/store/samples_generated --num_fid_samples=50000 --batch_size 64
```

More information can be found by running:
```
python modelzoo/models/vision/dit/sample_generator_dit.py -h
usage: sample_generator_dit.py [-h] [--seed SEED] [--model_ckpt_path MODEL_CKPT_PATH] [--vae_ckpt_path VAE_CKPT_PATH]
                               --params PARAMS [--variant VARIANT] [--num_fid_samples NUM_FID_SAMPLES] --sample_dir
                               SAMPLE_DIR [--batch_size BATCH_SIZE] [--create_grid]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED
  --model_ckpt_path MODEL_CKPT_PATH
                        Optional path to a diffusion model checkpoint (default: None)
  --vae_ckpt_path VAE_CKPT_PATH
                        Optional VAE model checkpoint path (default: None)
  --params PARAMS       Path to params to initialize Diffusion model and VAE models (default: None)
  --variant VARIANT     Variant of Diffusion model (default: None)
  --num_fid_samples NUM_FID_SAMPLES
                        number of samples to generate (default: 50000)
  --sample_dir SAMPLE_DIR
                        Directory to store generated samples (default: None)
  --batch_size BATCH_SIZE
                        per-gpu batch size for forward pass (default: None)
  --create_grid         If passed, create a grid from images generated (default: False)
```
**The script generates a `.npz` file that should be passed as input to FID score computation. Sample output looks as below:**
```
2023-09-06 15:55:30,585 INFO[sample_generator.py:49] Saved .npz file to /path/to/store/samples_generated/sample.npz [shape=(`num_fid_samples`, train_input.image_shape[0], train_input.image_shape[1], train_input.image_channels
)].
```
To generate samples belonging to specific ImageNet label classes, please set `model.reverse_params.pipeline.custom_labels` to a list of integer ImageNet labels. This will generate samples belonging to only these classes. For ex: if `model.reverse_params.pipeline.custom_labels: [207, 360]`, then we will only generate samples belonging to `golden_retriever`(label_id=207) and `otter`(label_id=360) classes respectively. The class ID and name corresponding to ImageNet labels can be found below:

<details>
  <summary>ImageNet ID  and label, expand this</summary>
    
    0 -- tench, Tinca tinca
    1 -- goldfish, Carassius auratus
    2 -- great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias
    3 -- tiger shark, Galeocerdo cuvieri
    4 -- hammerhead, hammerhead shark
    5 -- electric ray, crampfish, numbfish, torpedo
    6 -- stingray
    7 -- cock
    8 -- hen
    9 -- ostrich, Struthio camelus
    10 -- brambling, Fringilla montifringilla
    11 -- goldfinch, Carduelis carduelis
    12 -- house finch, linnet, Carpodacus mexicanus
    13 -- junco, snowbird
    14 -- indigo bunting, indigo finch, indigo bird, Passerina cyanea
    15 -- robin, American robin, Turdus migratorius
    16 -- bulbul
    17 -- jay
    18 -- magpie
    19 -- chickadee
    20 -- water ouzel, dipper
    21 -- kite
    22 -- bald eagle, American eagle, Haliaeetus leucocephalus
    23 -- vulture
    24 -- great grey owl, great gray owl, Strix nebulosa
    25 -- European fire salamander, Salamandra salamandra
    26 -- common newt, Triturus vulgaris
    27 -- eft
    28 -- spotted salamander, Ambystoma maculatum
    29 -- axolotl, mud puppy, Ambystoma mexicanum
    30 -- bullfrog, Rana catesbeiana
    31 -- tree frog, tree-frog
    32 -- tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui
    33 -- loggerhead, loggerhead turtle, Caretta caretta
    34 -- leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea
    35 -- mud turtle
    36 -- terrapin
    37 -- box turtle, box tortoise
    38 -- banded gecko
    39 -- common iguana, iguana, Iguana iguana
    40 -- American chameleon, anole, Anolis carolinensis
    41 -- whiptail, whiptail lizard
    42 -- agama
    43 -- frilled lizard, Chlamydosaurus kingi
    44 -- alligator lizard
    45 -- Gila monster, Heloderma suspectum
    46 -- green lizard, Lacerta viridis
    47 -- African chameleon, Chamaeleo chamaeleon
    48 -- Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis
    49 -- African crocodile, Nile crocodile, Crocodylus niloticus
    50 -- American alligator, Alligator mississipiensis
    51 -- triceratops
    52 -- thunder snake, worm snake, Carphophis amoenus
    53 -- ringneck snake, ring-necked snake, ring snake
    54 -- hognose snake, puff adder
    66 -- sand viper, horned viper, cerastes, horned asp, Cerastes cornutus
    55 -- green snake
    57 -- grass snake, garter snake
    56 -- king snake, kingsnake
    58 -- water snake
    59 -- vine snake
    60 -- night snake, Hypsiglena torquata
    61 -- boa constrictor, Constrictor constrictor
    62 -- rock python, rock snake, Python sebae
    63 -- Indian cobra, Naja naja
    64 -- green mamba
    65 -- sea snake
    67 -- diamondback, diamondback rattlesnake, Crotalus adamanteus
    68 -- sidewinder, horned rattlesnake, Crotalus cerastes
    69 -- trilobite
    70 -- harvestman, daddy longlegs, Phalangium opilio
    71 -- scorpion
    72 -- black and gold garden spider, Argiope aurantia
    73 -- barn spider, Araneus cavaticus
    74 -- garden spider, Aranea diademata
    75 -- black widow, Latrodectus mactans
    76 -- tarantula
    77 -- wolf spider, hunting spider
    78 -- tick
    79 -- centipede
    80 -- black grouse
    81 -- ptarmigan
    82 -- ruffed grouse, Bonasa umbellus
    86 -- partridge
    83 -- prairie chicken, prairie grouse, prairie fowl
    84 -- peacock
    85 -- quail
    87 -- African grey, African gray, Psittacus erithacus
    88 -- macaw
    89 -- sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
    90 -- lorikeet
    91 -- coucal
    92 -- bee eater
    93 -- hornbill
    94 -- hummingbird
    95 -- jacamar
    96 -- toucan
    97 -- drake
    98 -- red-breasted merganser, Mergus serrator
    99 -- goose
    100 -- black swan, Cygnus atratus
    101 -- tusker
    102 -- echidna, spiny anteater, anteater
    103 -- platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus
    104 -- wallaby, brush kangaroo
    105 -- koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
    106 -- wombat
    107 -- jellyfish
    108 -- sea anemone, anemone
    109 -- brain coral
    110 -- flatworm, platyhelminth
    111 -- nematode, nematode worm, roundworm
    112 -- conch
    113 -- snail
    114 -- slug
    115 -- sea slug, nudibranch
    116 -- chiton, coat-of-mail shell, sea cradle, polyplacophore
    117 -- chambered nautilus, pearly nautilus, nautilus
    118 -- Dungeness crab, Cancer magister
    119 -- rock crab, Cancer irroratus
    120 -- fiddler crab
    121 -- king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica
    122 -- American lobster, Northern lobster, Maine lobster, Homarus americanus
    123 -- spiny lobster, langouste, rock lobster, sea crawfish
    124 -- crawfish, crayfish, crawdad, crawdaddy
    125 -- hermit crab
    126 -- isopod
    127 -- white stork, Ciconia ciconia
    128 -- black stork, Ciconia nigra
    129 -- spoonbill
    130 -- flamingo
    131 -- little blue heron, Egretta caerulea
    132 -- American egret, great white heron, Egretta albus
    133 -- bittern
    517 -- crane
    135 -- limpkin, Aramus pictus
    136 -- European gallinule, Porphyrio porphyrio
    137 -- American coot, marsh hen, mud hen, water hen, Fulica americana
    138 -- bustard
    139 -- ruddy turnstone, Arenaria interpres
    140 -- red-backed sandpiper, dunlin, Erolia alpina
    141 -- redshank, Tringa totanus
    142 -- dowitcher
    143 -- oystercatcher, oyster catcher
    144 -- pelican
    145 -- king penguin, Aptenodytes patagonica
    146 -- albatross, mollymawk
    147 -- grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus
    148 -- killer whale, killer, orca, grampus, sea wolf, Orcinus orca
    149 -- dugong, Dugong dugon
    150 -- sea lion
    151 -- Chihuahua
    152 -- Japanese spaniel
    153 -- Maltese dog, Maltese terrier, Maltese
    154 -- Pekinese, Pekingese, Peke
    155 -- Shih-Tzu
    156 -- Blenheim spaniel
    157 -- papillon
    158 -- toy terrier
    159 -- Rhodesian ridgeback
    160 -- Afghan hound, Afghan
    161 -- basset, basset hound
    162 -- beagle
    163 -- bloodhound, sleuthhound
    164 -- bluetick
    165 -- black-and-tan coonhound
    166 -- Walker hound, Walker foxhound
    167 -- English foxhound
    168 -- redbone
    169 -- borzoi, Russian wolfhound
    170 -- Irish wolfhound
    171 -- Italian greyhound
    172 -- whippet
    173 -- Ibizan hound, Ibizan Podenco
    174 -- Norwegian elkhound, elkhound
    175 -- otterhound, otter hound
    176 -- Saluki, gazelle hound
    177 -- Scottish deerhound, deerhound
    178 -- Weimaraner
    179 -- Staffordshire bullterrier, Staffordshire bull terrier
    180 -- American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
    181 -- Bedlington terrier
    182 -- Border terrier
    183 -- Kerry blue terrier
    184 -- Irish terrier
    185 -- Norfolk terrier
    186 -- Norwich terrier
    187 -- Yorkshire terrier
    188 -- wire-haired fox terrier
    189 -- Lakeland terrier
    190 -- Sealyham terrier, Sealyham
    191 -- Airedale, Airedale terrier
    192 -- cairn, cairn terrier
    193 -- Australian terrier
    194 -- Dandie Dinmont, Dandie Dinmont terrier
    195 -- Boston bull, Boston terrier
    196 -- miniature schnauzer
    197 -- giant schnauzer
    198 -- standard schnauzer
    199 -- Scotch terrier, Scottish terrier, Scottie
    200 -- Tibetan terrier, chrysanthemum dog
    201 -- silky terrier, Sydney silky
    202 -- soft-coated wheaten terrier
    203 -- West Highland white terrier
    204 -- Lhasa, Lhasa apso
    205 -- flat-coated retriever
    206 -- curly-coated retriever
    207 -- golden retriever
    208 -- Labrador retriever
    209 -- Chesapeake Bay retriever
    210 -- German short-haired pointer
    211 -- vizsla, Hungarian pointer
    212 -- English setter
    213 -- Irish setter, red setter
    214 -- Gordon setter
    215 -- Brittany spaniel
    216 -- clumber, clumber spaniel
    217 -- English springer, English springer spaniel
    218 -- Welsh springer spaniel
    219 -- cocker spaniel, English cocker spaniel, cocker
    220 -- Sussex spaniel
    221 -- Irish water spaniel
    222 -- kuvasz
    223 -- schipperke
    224 -- groenendael
    225 -- malinois
    226 -- briard
    227 -- kelpie
    228 -- komondor
    229 -- Old English sheepdog, bobtail
    230 -- Shetland sheepdog, Shetland sheep dog, Shetland
    231 -- collie
    232 -- Border collie
    233 -- Bouvier des Flandres, Bouviers des Flandres
    234 -- Rottweiler
    235 -- German shepherd, German shepherd dog, German police dog, alsatian
    236 -- Doberman, Doberman pinscher
    237 -- miniature pinscher
    238 -- Greater Swiss Mountain dog
    239 -- Bernese mountain dog
    240 -- Appenzeller
    241 -- EntleBucher
    242 -- boxer
    243 -- bull mastiff
    244 -- Tibetan mastiff
    245 -- French bulldog
    246 -- Great Dane
    247 -- Saint Bernard, St Bernard
    248 -- Eskimo dog, husky
    249 -- malamute, malemute, Alaskan malamute
    250 -- Siberian husky
    251 -- dalmatian, coach dog, carriage dog
    252 -- affenpinscher, monkey pinscher, monkey dog
    253 -- basenji
    254 -- pug, pug-dog
    255 -- Leonberg
    256 -- Newfoundland, Newfoundland dog
    257 -- Great Pyrenees
    258 -- Samoyed, Samoyede
    259 -- Pomeranian
    260 -- chow, chow chow
    261 -- keeshond
    262 -- Brabancon griffon
    263 -- Pembroke, Pembroke Welsh corgi
    264 -- Cardigan, Cardigan Welsh corgi
    265 -- toy poodle
    266 -- miniature poodle
    267 -- standard poodle
    268 -- Mexican hairless
    269 -- timber wolf, grey wolf, gray wolf, Canis lupus
    270 -- white wolf, Arctic wolf, Canis lupus tundrarum
    271 -- red wolf, maned wolf, Canis rufus, Canis niger
    272 -- coyote, prairie wolf, brush wolf, Canis latrans
    273 -- dingo, warrigal, warragal, Canis dingo
    274 -- dhole, Cuon alpinus
    275 -- African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus
    276 -- hyena, hyaena
    277 -- red fox, Vulpes vulpes
    278 -- kit fox, Vulpes macrotis
    279 -- Arctic fox, white fox, Alopex lagopus
    280 -- grey fox, gray fox, Urocyon cinereoargenteus
    281 -- tabby, tabby cat
    282 -- tiger cat
    283 -- Persian cat
    284 -- Siamese cat, Siamese
    285 -- Egyptian cat
    286 -- cougar, puma, mountain lion, painter, Felis concolor
    287 -- catamount, lynx
    290 -- panther, jaguar, Panthera onca, Felis onca
    288 -- leopard, Panthera pardus
    289 -- snow leopard, ounce, Panthera uncia
    291 -- lion, king of beasts, Panthera leo
    292 -- tiger, Panthera tigris
    293 -- cheetah, chetah, Acinonyx jubatus
    294 -- brown bear, bruin, Ursus arctos
    295 -- American black bear, black bear, Ursus americanus, Euarctos americanus
    296 -- ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus
    297 -- sloth bear, Melursus ursinus, Ursus ursinus
    298 -- mongoose
    299 -- meerkat, mierkat
    300 -- tiger beetle
    301 -- ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
    302 -- ground beetle, carabid beetle
    303 -- long-horned beetle, longicorn, longicorn beetle
    304 -- leaf beetle, chrysomelid
    305 -- dung beetle
    306 -- rhinoceros beetle
    307 -- weevil
    308 -- fly
    309 -- bee
    310 -- ant, emmet, pismire
    311 -- grasshopper, hopper
    312 -- cricket
    313 -- walking stick, walkingstick, stick insect
    314 -- cockroach, roach
    315 -- mantis, mantid
    316 -- cicada, cicala
    317 -- leafhopper
    318 -- lacewing, lacewing fly
    319 -- dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk
    320 -- damselfly
    321 -- admiral
    322 -- ringlet, ringlet butterfly
    323 -- monarch, monarch butterfly, milkweed butterfly, Danaus plexippus
    324 -- cabbage butterfly
    325 -- sulphur butterfly, sulfur butterfly
    326 -- lycaenid, lycaenid butterfly
    327 -- starfish, sea star
    328 -- sea urchin
    329 -- sea cucumber, holothurian
    330 -- wood rabbit, cottontail, cottontail rabbit
    331 -- hare
    332 -- Angora, Angora rabbit
    333 -- hamster
    334 -- porcupine, hedgehog
    335 -- fox squirrel, eastern fox squirrel, Sciurus niger
    336 -- marmot
    337 -- beaver
    338 -- guinea pig, Cavia cobaya
    339 -- sorrel
    340 -- zebra
    341 -- hog, pig, grunter, squealer
    342 -- Sus scrofa, wild boar, boar
    343 -- warthog
    344 -- hippopotamus, hippo, river horse, Hippopotamus amphibius
    345 -- ox
    346 -- water buffalo, water ox, Asiatic buffalo, Bubalus bubalis
    347 -- bison
    348 -- ram, tup
    349 -- bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis
    350 -- ibex, Capra ibex
    351 -- hartebeest
    352 -- impala, Aepyceros melampus
    353 -- gazelle
    354 -- Arabian camel, dromedary, Camelus dromedarius
    355 -- llama
    356 -- weasel
    357 -- mink
    361 -- polecat, skunk, wood pussy
    358 -- fitch, foulmart, foumart, Mustela putorius
    359 -- black-footed ferret, ferret, Mustela nigripes
    360 -- otter
    362 -- badger
    363 -- armadillo
    364 -- three-toed sloth, ai, Bradypus tridactylus
    365 -- orangutan, orang, orangutang, Pongo pygmaeus
    366 -- gorilla, Gorilla gorilla
    367 -- chimpanzee, chimp, Pan troglodytes
    368 -- gibbon, Hylobates lar
    369 -- siamang, Hylobates syndactylus, Symphalangus syndactylus
    370 -- guenon, guenon monkey
    371 -- patas, hussar monkey, Erythrocebus patas
    372 -- baboon
    373 -- macaque
    374 -- langur
    375 -- colobus, colobus monkey
    376 -- proboscis monkey, Nasalis larvatus
    377 -- marmoset
    378 -- capuchin, ringtail, Cebus capucinus
    379 -- howler monkey, howler
    380 -- titi, titi monkey
    381 -- spider monkey, Ateles geoffroyi
    382 -- squirrel monkey, Saimiri sciureus
    383 -- Madagascar cat, ring-tailed lemur, Lemur catta
    384 -- indri, indris, Indri indri, Indri brevicaudatus
    385 -- Indian elephant, Elephas maximus
    386 -- African elephant, Loxodonta africana
    387 -- lesser panda, red panda, bear cat, cat bear, Ailurus fulgens
    388 -- panda, giant panda, panda bear, coon bear, Ailuropoda melanoleuca
    389 -- barracouta, snoek
    390 -- eel
    391 -- coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch
    392 -- rock beauty, Holocanthus tricolor
    393 -- anemone fish
    394 -- sturgeon
    395 -- gar, garfish, garpike, billfish, Lepisosteus osseus
    396 -- lionfish
    397 -- puffer, pufferfish, blowfish, globefish
    398 -- abacus
    399 -- abaya
    400 -- academic gown, academic robe, judge's robe
    401 -- accordion, piano accordion, squeeze box
    402 -- acoustic guitar
    403 -- aircraft carrier, carrier, flattop, attack aircraft carrier
    404 -- airliner
    405 -- airship, dirigible
    406 -- altar
    407 -- ambulance
    408 -- amphibian, amphibious vehicle
    409 -- analog clock
    410 -- apiary, bee house
    411 -- apron
    412 -- ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin
    413 -- assault rifle, assault gun
    414 -- backpack, back pack, knapsack, packsack, rucksack, haversack
    415 -- bakery, bakeshop, bakehouse
    416 -- balance beam, beam
    417 -- balloon
    418 -- ballpoint, ballpoint pen, ballpen, Biro
    419 -- Band Aid
    420 -- banjo
    421 -- bannister, banister, balustrade, balusters, handrail
    422 -- barbell
    423 -- barber chair
    424 -- barbershop
    425 -- barn
    426 -- barometer
    427 -- barrel, cask
    428 -- barrow, garden cart, lawn cart, wheelbarrow
    429 -- baseball
    430 -- basketball
    431 -- bassinet
    432 -- bassoon
    433 -- bathing cap, swimming cap
    434 -- bath towel
    435 -- bathtub, bathing tub, bath
    876 -- tub, vat
    436 -- beach wagon, station wagon, estate car, beach waggon, station waggon, waggon
    734 -- wagon, police van, police wagon, paddy wagon, patrol wagon, black Maria
    437 -- beacon, lighthouse, beacon light, pharos
    438 -- beaker
    439 -- bearskin, busby, shako
    440 -- beer bottle
    441 -- beer glass
    442 -- bell cote, bell cot
    443 -- bib
    444 -- bicycle-built-for-two, tandem bicycle, tandem
    445 -- bikini, two-piece
    446 -- binder, ring-binder
    447 -- binoculars, field glasses, opera glasses
    448 -- birdhouse
    449 -- boathouse
    450 -- bobsled, bobsleigh, bob
    451 -- bolo tie, bolo, bola tie, bola
    452 -- bonnet, poke bonnet
    453 -- bookcase
    454 -- bookshop, bookstore, bookstall
    455 -- bottlecap
    456 -- bow
    457 -- bow tie, bow-tie, bowtie
    458 -- brass, memorial tablet, plaque
    459 -- brassiere, bra, bandeau
    460 -- breakwater, groin, groyne, mole, bulwark, seawall, jetty
    461 -- breastplate, aegis, egis
    462 -- broom
    463 -- bucket, pail
    464 -- buckle
    465 -- bulletproof vest
    466 -- bullet train, bullet
    467 -- butcher shop, meat market
    468 -- cab, hack, taxi, taxicab
    469 -- caldron, cauldron
    470 -- candle, taper, wax light
    471 -- cannon
    472 -- canoe
    473 -- can opener, tin opener
    474 -- cardigan
    475 -- car mirror
    476 -- carousel, carrousel, merry-go-round, roundabout, whirligig
    477 -- carpenter's kit, tool kit
    478 -- carton
    479 -- car wheel
    480 -- cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM
    481 -- cassette
    482 -- cassette player
    483 -- castle
    484 -- catamaran
    485 -- CD player
    486 -- cello, violoncello
    487 -- cellular telephone, cellular phone, cellphone, cell, mobile phone
    488 -- chain
    489 -- chainlink fence
    490 -- chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour
    491 -- chain saw, chainsaw
    492 -- chest
    493 -- chiffonier, commode
    494 -- chime, bell
    577 -- gong, tam-tam
    495 -- china cabinet, china closet
    496 -- Christmas stocking
    497 -- church, church building
    498 -- cinema, movie theater, movie theatre, movie house, picture palace
    499 -- cleaver, meat cleaver, chopper
    500 -- cliff dwelling
    501 -- cloak
    502 -- clog, geta, patten, sabot
    503 -- cocktail shaker
    504 -- coffee mug
    505 -- coffeepot
    506 -- coil, spiral, volute, whorl, helix
    507 -- combination lock
    508 -- computer keyboard, keypad
    509 -- confectionery, confectionary, candy store
    510 -- container ship, containership, container vessel
    511 -- convertible
    512 -- corkscrew, bottle screw
    513 -- cornet, trumpet, trump
    566 -- horn, French horn
    514 -- cowboy boot
    515 -- cowboy hat, ten-gallon hat
    516 -- cradle
    518 -- crash helmet
    519 -- crate
    520 -- crib, cot
    521 -- Crock Pot
    522 -- croquet ball
    523 -- crutch
    524 -- cuirass
    525 -- dam, dike, dyke
    526 -- desk
    527 -- desktop computer
    528 -- dial telephone, dial phone
    529 -- diaper, nappy, napkin
    530 -- digital clock
    531 -- digital watch
    532 -- dining table, board
    533 -- dishrag, dishcloth
    534 -- dishwasher, dish washer, dishwashing machine
    535 -- disk brake, disc brake
    536 -- dock, dockage, docking facility
    537 -- dogsled, dog sled, dog sleigh
    538 -- dome
    539 -- doormat, welcome mat
    540 -- drilling platform, offshore rig
    541 -- drum, membranophone, tympan
    542 -- drumstick
    543 -- dumbbell
    544 -- Dutch oven
    545 -- electric fan, blower
    546 -- electric guitar
    547 -- electric locomotive
    548 -- entertainment center
    549 -- envelope
    550 -- espresso maker
    551 -- face powder
    552 -- feather boa, boa
    553 -- file, file cabinet, filing cabinet
    554 -- fireboat
    555 -- fire engine, fire truck
    556 -- fire screen, fireguard
    557 -- flagpole, flagstaff
    558 -- flute, transverse flute
    559 -- folding chair
    560 -- football helmet
    561 -- forklift
    562 -- fountain
    563 -- fountain pen
    564 -- four-poster
    565 -- freight car
    567 -- frying pan, frypan, skillet
    568 -- fur coat
    569 -- garbage truck, dustcart
    570 -- gasmask, respirator, gas helmet
    571 -- gas pump, gasoline pump, petrol pump, island dispenser
    572 -- goblet
    573 -- go-kart
    574 -- golf ball
    575 -- golfcart, golf cart
    576 -- gondola
    578 -- gown
    579 -- grand piano, grand
    580 -- greenhouse, nursery, glasshouse
    581 -- grille, radiator grille
    582 -- grocery store, grocery, food market, market
    583 -- guillotine
    584 -- hair slide
    585 -- hair spray
    586 -- half track
    587 -- hammer
    588 -- hamper
    589 -- hand blower, blow dryer, blow drier, hair dryer, hair drier
    590 -- hand-held computer, hand-held microcomputer
    591 -- handkerchief, hankie, hanky, hankey
    592 -- hard disc, hard disk, fixed disk
    593 -- harmonica, mouth organ, mouth harp
    594 -- harp
    595 -- harvester, reaper
    596 -- hatchet
    597 -- holster
    598 -- home theater, home theatre
    599 -- honeycomb
    600 -- hook, claw
    601 -- hoopskirt, crinoline
    602 -- horizontal bar, high bar
    603 -- horse cart, horse-cart
    604 -- hourglass
    605 -- iPod
    606 -- iron, smoothing iron
    607 -- jack-o'-lantern
    608 -- jean, blue jean, denim
    609 -- jeep, landrover
    610 -- jersey, T-shirt, tee shirt
    611 -- jigsaw puzzle
    612 -- jinrikisha, ricksha, rickshaw
    613 -- joystick
    614 -- kimono
    615 -- knee pad
    616 -- knot
    617 -- lab coat, laboratory coat
    618 -- ladle
    619 -- lampshade, lamp shade
    620 -- laptop, laptop computer
    621 -- lawn mower, mower
    622 -- lens cap, lens cover
    623 -- letter opener, paper knife, paperknife
    624 -- library
    625 -- lifeboat
    626 -- lighter, light, igniter, ignitor
    627 -- limousine, limo
    628 -- liner, ocean liner
    629 -- lipstick, lip rouge
    630 -- Loafer
    631 -- lotion
    632 -- loudspeaker, speaker, speaker unit, loudspeaker system, speaker system
    633 -- loupe, jeweler's loupe
    634 -- lumbermill, sawmill
    635 -- magnetic compass
    636 -- mailbag, postbag
    637 -- mailbox, letter box
    639 -- maillot, tank suit
    640 -- manhole cover
    641 -- maraca
    642 -- marimba, xylophone
    643 -- mask
    644 -- matchstick
    645 -- maypole
    646 -- maze, labyrinth
    647 -- measuring cup
    648 -- medicine chest, medicine cabinet
    649 -- megalith, megalithic structure
    650 -- microphone, mike
    651 -- microwave, microwave oven
    652 -- military uniform
    653 -- milk can
    654 -- minibus
    655 -- miniskirt, mini
    656 -- minivan
    744 -- missile, projectile
    658 -- mitten
    659 -- mixing bowl
    660 -- mobile home, manufactured home
    661 -- Model T
    662 -- modem
    663 -- monastery
    664 -- monitor
    665 -- moped
    666 -- mortar
    667 -- mortarboard
    668 -- mosque
    669 -- mosquito net
    670 -- motor scooter, scooter
    671 -- mountain bike, all-terrain bike, off-roader
    672 -- mountain tent
    673 -- mouse, computer mouse
    674 -- mousetrap
    675 -- moving van
    676 -- muzzle
    677 -- nail
    678 -- neck brace
    679 -- necklace
    680 -- nipple
    681 -- notebook, notebook computer
    682 -- obelisk
    683 -- oboe, hautboy, hautbois
    684 -- ocarina, sweet potato
    685 -- odometer, hodometer, mileometer, milometer
    686 -- oil filter
    687 -- organ, pipe organ
    688 -- oscilloscope, scope, cathode-ray oscilloscope, CRO
    689 -- overskirt
    690 -- oxcart
    691 -- oxygen mask
    692 -- packet
    693 -- paddle, boat paddle
    694 -- paddlewheel, paddle wheel
    695 -- padlock
    696 -- paintbrush
    697 -- pajama, pyjama, pj's, jammies
    698 -- palace
    699 -- panpipe, pandean pipe, syrinx
    700 -- paper towel
    701 -- parachute, chute
    702 -- parallel bars, bars
    703 -- park bench
    704 -- parking meter
    705 -- passenger car, coach, carriage
    706 -- patio, terrace
    707 -- pay-phone, pay-station
    708 -- pedestal, plinth, footstall
    709 -- pencil box, pencil case
    710 -- pencil sharpener
    711 -- perfume, essence
    712 -- Petri dish
    713 -- photocopier
    714 -- pick, plectrum, plectron
    715 -- pickelhaube
    716 -- picket fence, paling
    717 -- pickup, pickup truck
    718 -- pier
    719 -- piggy bank, penny bank
    720 -- pill bottle
    721 -- pillow
    722 -- ping-pong ball
    723 -- pinwheel
    724 -- pirate, pirate ship
    725 -- pitcher, ewer
    726 -- plane, carpenter's plane, woodworking plane
    727 -- planetarium
    728 -- plastic bag
    729 -- plate rack
    730 -- plow, plough
    731 -- plunger, plumber's helper
    732 -- Polaroid camera, Polaroid Land camera
    733 -- pole
    735 -- poncho
    736 -- pool table, billiard table, snooker table
    737 -- pop bottle, soda bottle
    738 -- pot, flowerpot
    739 -- potter's wheel
    740 -- power drill
    741 -- prayer rug, prayer mat
    742 -- printer
    743 -- prison, prison house
    745 -- projector
    746 -- puck, hockey puck
    747 -- punching bag, punch bag, punching ball, punchball
    748 -- purse
    749 -- quill, quill pen
    750 -- quilt, comforter, comfort, puff
    751 -- racer, race car, racing car
    752 -- racket, racquet
    753 -- radiator
    754 -- radio, wireless
    755 -- radio telescope, radio reflector
    756 -- rain barrel
    757 -- recreational vehicle, RV, R.V.
    758 -- reel
    759 -- reflex camera
    760 -- refrigerator, icebox
    761 -- remote control, remote
    762 -- restaurant, eating house, eating place, eatery
    763 -- revolver, six-gun, six-shooter
    764 -- rifle
    765 -- rocking chair, rocker
    766 -- rotisserie
    767 -- rubber eraser, rubber, pencil eraser
    768 -- rugby ball
    769 -- rule, ruler
    770 -- running shoe
    771 -- safe
    772 -- safety pin
    773 -- saltshaker, salt shaker
    774 -- sandal
    775 -- sarong
    776 -- sax, saxophone
    777 -- scabbard
    778 -- scale, weighing machine
    779 -- school bus
    780 -- schooner
    781 -- scoreboard
    782 -- screen, CRT screen
    783 -- screw
    784 -- screwdriver
    785 -- seat belt, seatbelt
    786 -- sewing machine
    787 -- shield, buckler
    788 -- shoe shop, shoe-shop, shoe store
    789 -- shoji
    790 -- shopping basket
    791 -- shopping cart
    792 -- shovel
    793 -- shower cap
    794 -- shower curtain
    795 -- ski
    796 -- ski mask
    797 -- sleeping bag
    798 -- slide rule, slipstick
    799 -- sliding door
    800 -- slot, one-armed bandit
    801 -- snorkel
    802 -- snowmobile
    803 -- snowplow, snowplough
    804 -- soap dispenser
    805 -- soccer ball
    806 -- sock
    807 -- solar dish, solar collector, solar furnace
    808 -- sombrero
    809 -- soup bowl
    810 -- space bar
    811 -- space heater
    812 -- space shuttle
    813 -- spatula
    814 -- speedboat
    815 -- spider web, spider's web
    816 -- spindle
    817 -- sports car, sport car
    818 -- spotlight, spot
    819 -- stage
    820 -- steam locomotive
    821 -- steel arch bridge
    822 -- steel drum
    823 -- stethoscope
    824 -- stole
    825 -- stone wall
    826 -- stopwatch, stop watch
    827 -- stove
    828 -- strainer
    829 -- streetcar, tram, tramcar, trolley, trolley car
    830 -- stretcher
    831 -- studio couch, day bed
    832 -- stupa, tope
    833 -- submarine, pigboat, sub, U-boat
    834 -- suit, suit of clothes
    835 -- sundial
    836 -- sunglass
    837 -- sunglasses, dark glasses, shades
    838 -- sunscreen, sunblock, sun blocker
    839 -- suspension bridge
    840 -- swab, swob, mop
    841 -- sweatshirt
    842 -- swimming trunks, bathing trunks
    843 -- swing
    844 -- switch, electric switch, electrical switch
    845 -- syringe
    846 -- table lamp
    847 -- tank, army tank, armored combat vehicle, armoured combat vehicle
    848 -- tape player
    849 -- teapot
    850 -- teddy, teddy bear
    851 -- television, television system
    852 -- tennis ball
    853 -- thatch, thatched roof
    854 -- theater curtain, theatre curtain
    855 -- thimble
    856 -- thresher, thrasher, threshing machine
    857 -- throne
    858 -- tile roof
    859 -- toaster
    860 -- tobacco shop, tobacconist shop, tobacconist
    861 -- toilet seat
    862 -- torch
    863 -- totem pole
    864 -- tow truck, tow car, wrecker
    865 -- toyshop
    866 -- tractor
    867 -- trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi
    868 -- tray
    869 -- trench coat
    870 -- tricycle, trike, velocipede
    871 -- trimaran
    872 -- tripod
    873 -- triumphal arch
    874 -- trolleybus, trolley coach, trackless trolley
    875 -- trombone
    877 -- turnstile
    878 -- typewriter keyboard
    879 -- umbrella
    880 -- unicycle, monocycle
    881 -- upright, upright piano
    882 -- vacuum, vacuum cleaner
    883 -- vase
    884 -- vault
    885 -- velvet
    886 -- vending machine
    887 -- vestment
    888 -- viaduct
    889 -- violin, fiddle
    890 -- volleyball
    891 -- waffle iron
    892 -- wall clock
    893 -- wallet, billfold, notecase, pocketbook
    894 -- wardrobe, closet, press
    895 -- warplane, military plane
    896 -- washbasin, handbasin, washbowl, lavabo, wash-hand basin
    897 -- washer, automatic washer, washing machine
    898 -- water bottle
    899 -- water jug
    900 -- water tower
    901 -- whiskey jug
    902 -- whistle
    903 -- wig
    904 -- window screen
    905 -- window shade
    906 -- Windsor tie
    907 -- wine bottle
    908 -- wing
    909 -- wok
    910 -- wooden spoon
    911 -- wool, woolen, woollen
    912 -- worm fence, snake fence, snake-rail fence, Virginia fence
    913 -- wreck
    914 -- yawl
    915 -- yurt
    916 -- web site, website, internet site, site
    917 -- comic book
    918 -- crossword puzzle, crossword
    919 -- street sign
    920 -- traffic light, traffic signal, stoplight
    921 -- book jacket, dust cover, dust jacket, dust wrapper
    922 -- menu
    923 -- plate
    924 -- guacamole
    925 -- consomme
    926 -- hot pot, hotpot
    927 -- trifle
    928 -- ice cream, icecream
    929 -- ice lolly, lolly, lollipop, popsicle
    930 -- French loaf
    931 -- bagel, beigel
    932 -- pretzel
    933 -- cheeseburger
    934 -- hotdog, hot dog, red hot
    935 -- mashed potato
    936 -- head cabbage
    937 -- broccoli
    938 -- cauliflower
    939 -- zucchini, courgette
    940 -- spaghetti squash
    941 -- acorn squash
    942 -- butternut squash
    943 -- cucumber, cuke
    944 -- artichoke, globe artichoke
    945 -- bell pepper
    946 -- cardoon
    947 -- mushroom
    948 -- Granny Smith
    949 -- strawberry
    950 -- orange
    951 -- lemon
    952 -- fig
    953 -- pineapple, ananas
    954 -- banana
    955 -- jackfruit, jak, jack
    956 -- custard apple
    957 -- pomegranate
    958 -- hay
    959 -- carbonara
    960 -- chocolate sauce, chocolate syrup
    961 -- dough
    962 -- meat loaf, meatloaf
    963 -- pizza, pizza pie
    964 -- potpie
    965 -- burrito
    966 -- red wine
    967 -- espresso
    968 -- cup
    969 -- eggnog
    970 -- alp
    971 -- bubble
    972 -- cliff, drop, drop-off
    973 -- coral reef
    974 -- geyser
    975 -- lakeside, lakeshore
    976 -- promontory, headland, head, foreland
    977 -- sandbar, sand bar
    978 -- seashore, coast, seacoast, sea-coast
    979 -- valley, vale
    980 -- volcano
    981 -- ballplayer, baseball player
    982 -- groom, bridegroom
    983 -- scuba diver
    984 -- rapeseed
    985 -- daisy
    986 -- yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum
    987 -- corn
    988 -- acorn
    989 -- hip, rose hip, rosehip
    990 -- buckeye, horse chestnut, conker
    991 -- coral fungus
    992 -- agaric
    993 -- gyromitra
    994 -- stinkhorn, carrion fungus
    995 -- earthstar
    996 -- hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa
    997 -- bolete
    998 -- ear, spike, capitulum
    999 -- toilet tissue, toilet paper, bathroom tissue
    
</details>

### Step 6: Using OpenAI FID evaluation repository to compute FID score

Now that we have the 50K samples and `.npz` file generated from Step 5, we can compute FID score using ADM OpenAI script [evaluator.py](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py). 
In order to compute FID score, 

#### a. Set up a conda environment to use OpenAI evaluation script
```
conda create --name tf python=3.8.16
conda activate tf
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
pip install --upgrade pip
pip install tensorflow==2.13.*
conda install scipy
conda install -c conda-forge tqdm
conda install -c anaconda requests
conda install -c anaconda chardet
```

#### b. Clone [OpenAI guided-diffusion GitHub repository](https://github.com/openai/guided-diffusion/tree/main)
```
git clone https://github.com/openai/guided-diffusion.git
cd guided-diffusion/evaluations
```

#### c. Download the `npz` files corresponding to reference batch of ImageNet
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz -P /path/to/store/reference_batch
``` 

#### d. Make changes to `evaluator.py`
Make the following changes in [evaluator.py](https://github.com/openai/guided-diffusion/blob/main/evaluations/evaluator.py). These are needed to account for [numpy deprecations](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations) i.e replace instances of `np.bool` with bool
```
[evaluations](main)$ git diff
diff --git a/evaluations/evaluator.py b/evaluations/evaluator.py
index 9590855..6636d0b 100644
--- a/evaluations/evaluator.py
+++ b/evaluations/evaluator.py
@@ -340,8 +340,8 @@ class ManifoldEstimator:
                - precision: an np.ndarray of length K1
                - recall: an np.ndarray of length K2
        """
-        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=np.bool)
-        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=np.bool)
+        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
+        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
```

#### e. Launch FID eval script with the following command
```    
conda activate tf
cd guided-diffusion/evaluations
python evaluator.py  /path/to/step(d)/downloaded/VIRTUAL_imagenet256_labeled.npz /path/to/generated/npz/from/step5 2>&1 | tee fid.log
```   
   


## Configuration files included for this model

We provide the following config files for DiT located under the [configs](configs) directory.

| Config File | Dataset | Data Processor | Block Type | Patch Size | Image Size | Latent Size | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [params_dit_small_patchsize_2x2.yaml](./configs/params_dit_small_patchsize_2x2.yaml) | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) |[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) | `adaln_zero` | 2 x 2 | 256 x 256 x 3 | 32 x 32 x 4 | DiT-S/2 with ~33M params |
| [params_dit_base_patchsize_2x2.yaml](./configs/params_dit_base_patchsize_2x2.yaml) | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) |[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) | `adaln_zero` | 2 x 2 | 256 x 256 x 3 | 32 x 32 x 4 | DiT-B/2 with ~130M params |
| [params_dit_large_patchsize_2x2.yaml](./configs/params_dit_large_patchsize_2x2.yaml) | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) |[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) | `adaln_zero` | 2 x 2 | 256 x 256 x 3 | 32 x 32 x 4 | DiT-L/2 with ~458M params |
| [params_dit_xlarge_patchsize_2x2.yaml](./configs/params_dit_xlarge_patchsize_2x2.yaml) | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) |[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) | `adaln_zero` | 2 x 2 | 256 x 256 x 3 | 32 x 32 x 4 | DiT-XL/2 with ~675M params |
| [params_dit_2B_patchsize_2x2.yaml](./configs/params_dit_2B_patchsize_2x2.yaml) | [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) |[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) | `adaln_zero` | 2 x 2 | 256 x 256 x 3 | 32 x 32 x 4 | DiT-2B/2 with ~2B params |


The following changes can be made to use other settings of DiT model:
-   The `model.vae.latent_size` parameter corresponds size of latent tensors. **This is the only param under `model.vae_params` that needs to be changed.**
    -  Set to `[32, 32]` for image size of `256 x 256`
    -  Set to `[64, 64]` for image size of `512 x 512`
    -  Set to `[floor(H / 8), floor(W / 8)]` for image size of `H x W`
-   The `model.patch_size` parameter to use different patch sizes

## DataLoader Features Dictionary

[DiffusionLatentImageNet1KProcessor](../../../data/vision/diffusion/DiffusionLatentImageNet1KProcessor.py) outputs the following features dictionary with keys/values:

- `input`: Noised latent tensor.
  - Shape: `(batch_size, model.vae.latent_channels, model.vae.latent_size[0], model.vae.latent_size[1])`
  - Type: `torch.bfloat16`
- `label`: Scalar ImageNet labels.
  - Shape: `(batch_size, )`
  - Type: `torch.int32`
- `diffusion_noise`: Gaussian noise that the model should predict. Also used in creating value of key `noised_latent`.
  - Shape: `(batch_size, model.vae.latent_channels, model.vae.latent_size[0], model.vae.latent_size[1])`
  - Type: `torch.bfloat16`
- `timestep`: Timestep sampled from _~Uniform(0, `train_input.num_diffusion_steps`)_.
  - Shape: `(batch_size, )`
  - Type: `torch.int32`


## Implementation notes

There are a couple modifications to the DiT model made in this implementation:

1. We use `ConvTranspose2D` instead of `Linear` layer to un-patchify the outputs.
2. While we support `gelu with approximation tanh`, we use `gelu with no approximation` for better performance.
3. Inorder to use the exact model as [StabilityAI pretrained VAE model](https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin), we don't have to make any changes to the params under `model.vae_params`. The only modification we make in our implementation of VAE Model is that we use [Attention Layer defined in modelzoo](../../../layers/AttentionLayer.py). 
4. We currently do not support Kullback-Leibler(KL) loss to optimize Σ, hence the output from DiT Model includes only the noise.
5. We currently support `AdaLN-Zero` variant of DiT model. Support for `In-Context` and `Cross-Attention` variants are planned for future releases.


## Citations

[1] [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748.pdf)

[2] [Denoising Diffusion Probabilistic Models](https://hojonathanho.github.io/diffusion/assets/denoising_diffusion20.pdf)

[3] [Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502.pdf)

[4] [ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/challenges/LSVRC/2012/)

[5] [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

[6] [Guided Diffusion GitHub](https://github.com/openai/guided-diffusion)

[7] [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf)

[8] [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)