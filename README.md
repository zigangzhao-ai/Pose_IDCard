# Pose_IDCard

## Introduction


## Dependencies
* [TensorFlow](https://www.tensorflow.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* [COCO API](https://github.com/cocodataset/cocoapi)

This code is trained under Ubuntu 16.04, CUDA 10.0, cuDNN 7.1 environment with one NVIDIA 1080Ti GPUs.
Python 3.6.5 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- data
|-- lib
|-- main
|-- tes_coco
|-- scripts
|-- main
`-- output
```

### Data
You need to follow directory structure of the `data` as below.
```
${POSE_ROOT}
|-- data
|-- |-- MPII
|       |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg

`-- |-- imagenet_weights
|       |-- mobilenetv2_1.4_224.ckpt
```


### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- log
|-- |-- model_dump
|-- |-- result
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.
* You can change default directory structure of `output` by modifying `main/config.py`.

## Running Pose_IDCard
### Start
* Run `pip install -r requirement.txt` to install required modules.
* Run `cd ${POSE_ROOT}/lib` and `make` to build NMS modules.
* In the `main/config.py`, you can change settings of the model including dataset to use, network backbone, and input size and so on.

### Data Preprocessing 
* We can convert own dataset to [MS COCO format](http://cocodataset.org/#format-data).
* In the `tool`, run `json_to_txt.py` to convert original annotation files to txt format .
* In the `tool`, run `txt_to_xml.py` to convert txt annotation files to VOC xml format.
* In the `tool`, run `xml_to_coco.py` to convert xml annotation files to COCO json format.

* Download imagenet pre-trained mobilenetv2 models from [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim) and place it in the `data/imagenet_weights`.


### Train
In the `main` folder, run
```bash
python train.py --gpu 0-1
```
to train the network on the GPU 0,1. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0-1 --continue
```
`--gpu 0,1` can be used instead of `--gpu 0-1`.

### Test
Place trained model at the `output/model_dump/$DATASET/` and human detection result (`human_detection.json`) to `data/$DATASET/dets/`.

In the `main` folder, run 
```bash
python test.py --gpu 0-1 --test_epoch 180
```
to test the network on the GPU 0,1 with 180th epoch trained model. `--gpu 0,1` can be used instead of `--gpu 0-1`.

If you want to test a single image, run 
```bash
python test_one.py --gpu 0-1
```

## Results

### Results on val_dataset

| Methods | AP .5 | AP .75 | Mean_distance| 
|:---:|:---:|:---:|:---:|:---:|
|256x192_mobilenetv2_1.4_224<br>(t) | 96.85 | 90.50 | 305.74 | 
|384x288_mobilenetv2_1.4_224<br>(t)| 96.85 | 87.5 | 316.13 | 
|256x256_mobilenetv2_1.4_224<br>(t)| 95.27| 88.18 | 317.08 | 
|256x192_mobilenetv2_1.4_224_fpn<br>(t) | 96.85 | 93.70 | 307.76 | 
|256x192_mobilenetv2_1.0_224fpn<br>(t) | 99.21 | 96.85 | 321.42 | 

## Reference
[1] Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple Baselines for Human Pose Estimation and Tracking". ECCV 2018.
[2]. https://github.com/mks0601/TF-SimpleHumanPose
