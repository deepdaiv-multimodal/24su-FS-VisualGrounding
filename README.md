
# Dynamic FS-MDETR: A Dynamic Multimodal Transformer Decoder for Visual Grounding

## Overview

Dynamic FS-MDETR is a multimodal transformer model designed for few-shot visual grounding tasks. This repository provides instructions for setting up the project environment, training the model, and preparing the dataset.

## Table of Contents

1. [Environment Setup for Main Project](#environment-setup-for-main-project)
2. [Training the Model](#training-the-model)
3. [Inference](#inference)
4. [* Environment Setup for Building Datasets *](#environment-setup-for-building-datasets)
5. [* Preparing Datasets* ](#preparing-datasets)

## Environment Setup for Main Project

To set up the environment for the Dynamic MDETR model, follow these steps:

1. **Create a Conda environment:**

   ```bash
   conda create -n dynamic-fsmdetr python=3.10
   conda activate dynamic-fsmdetr
   ```

2. **Install dependencies:**

   Run the installation script:

   ```bash
   bash install.txt
   ```

## Training the Model

After setting up the environment, you can start training the model. Use the following commands depending on the dataset:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Training for different datasets
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --model_type ResNet --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --clip_max_norm 1.0
# (Repeat for other datasets as shown in the original guide)
```

## Inference

To evaluate the model, use the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --model_type ResNet --batch_size 16 --backbone resnet50 --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --stages 3 --vl_fusion_enc_layers 3 --uniform_learnable True --in_points 36 --lr 1e-4 --different_transformer True --lr_drop 60 --vl_dec_layers 1 --vl_enc_layers 1 --eval_model outputs/refcocog_gsplit_r50/best_checkpoint.pth --eval_set val
```

## Environment Setup for Building Datasets

To build the dataset environment separately, follow these steps:

1. **Create a new Conda environment:**

   ```bash
   conda create --name groundvlp python=3.8
   conda activate groundvlp
   ```

2. **Clone the GroundVLP repository:**

   ```bash
   git clone https://github.com/om-ai-lab/GroundVLP.git
   cd GroundVLP
   ```

3. **Install PyTorch and Detectron2:**

   ```bash
   pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
   python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
   ```

4. **Install additional dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Preparing Datasets

1. **Download and place the required checkpoints:**

   Download the following checkpoints and place them in the `checkpoints/` directory:
   - [ALBEF](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth)
   - [TCL](https://drive.google.com/file/d/1Cb1azBdcdbm0pRMFs-tupKxILTCXlB4O/view)
   - [Detic](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth)

2. **Download JSON files for RefCOCO/+/g:**

   Download and unzip them in the `data/` directory:
   - [RefCOCO/+/g INFO Json files](https://drive.google.com/file/d/1IPACy7Tb1XAK_uWGSXGDZrY-4txCOhSG/view?usp=sharing)

3. **Download COCO images:**

   Download the COCO images and unzip them in the `images/train2014` directory:
   - [~~COCO_train2014~~](http://images.cocodataset.org/zips/train2014.zip)
   
   (*This link has been broken. Use below instead.*)
   ```bash
   wget http://images.cocodataset.org/zips/train2014.zip
   ```

4. **Run dataset evaluation:**

   To evaluate GroundVLP on REC datasets using the ground-truth category, run:

   ```bash
   python eval_rec.py --image_folder="./images/train2014" --eval_data="refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test" --model_id="ALBEF" --use_gt_category
   ```

## Acknowledgments

This project is built upon [TransVG](https://github.com/djiajunustc/TransVG). Thanks for their wonderful work!
