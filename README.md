<span align="center">
<h1> OpenNeRF: OpenSet 3D Neural Scene Segmentation<br>with Pixel-Wise Features and Rendered Novel Views</h1>

<a href="https://francisengelmann.github.io">Francis Engelmann</a>,
<a href="https://scholar.google.de/citations?user=bERItx8AAAAJ">Fabian Manhardt</a>,
<a href="https://m-niemeyer.github.io">Michael Niemeyer</a>,
<a href="https://scholar.google.com/citations?user=ml3laqEAAAAJ">Keisuke Tateno</a>,
<a href="https://inf.ethz.ch/people/person-detail.pollefeys.html">Marc Pollefeys</a>,
<a href="https://federicotombari.github.io">Federico Tombari</a>

<h3>ICLR 2024</h3>

<a href="https://arxiv.org/abs/2404.03650">Paper</a> | <a href="http://opennerf.github.io">Project Page</a>

</span>

![OpenNeRF Teaser](https://opennerf.github.io/static/images/teaser.png)

## Installation

#### Install NerfStudio

After installing conda (see [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)), setup the conda environment:

```
conda create --name opennerf -y python=3.10
conda activate opennerf
python -m pip install --upgrade pip
```

### Install cuda, torch, etc.

```
conda install nvidia/label/cuda-12.1.1::cuda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install OpenNeRF

```
git clone https://github.com/opennerf/opennerf
cd opennerf
python -m pip install -e .
ns-install-cli
```

## Data preparation and OpenSeg Model

The datasets and saved NeRF models require significant disk space.
Let's link them to some (remote) larger storage:
```
ln -s path/to/large_disk/data data
ln -s path/to/large_disk/models models
ln -s path/to/large_disk/outputs outputs
```

Download the OpenSeg feature extractor model from [here](https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view?usp=sharing) and unzip it into `./models`.

### Replica Dataset
Download the Replica dataset pre-processed by [NICE-SLAM](https://pengsongyou.github.io/nice-slam) and transform it into [nerfstudio](https://docs.nerf.studio) format using these steps:
```
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
cd ..
python datasets/replica_preprocess.py
```

### LERF Dataset
The preprocessed LERF dataset is available from the [official repository](https://drive.google.com/drive/folders/1vh0mSl7v29yaGsxleadcj-LCZOE_WEWB).

## Running OpenNeRF

This repository creates a new Nerfstudio method named "opennerf". To train with it, run the command:
```
ns-train opennerf --data [PATH]
```
See `.vscode/launch.json` for specific training examples.

To view the optimized NeRF, you can launch the viewer separately:
```
ns-viewer --load-config outputs/path_to/config.yml
```

## Semantic Predictions

```TODO```

## ARKitScenes

```TODO```

## Coordinate Frames

Object coordinate frame: x - right, y - up, z - backwards (away from scene)

## File Structure

```
├── opennerf
│   ├── 
│   ├── __init__.py
│   ├── data
│   │   ├── utils
│   │   │   ├── dino_dataloader.py
│   │   │   ├── dino_extractor.py
│   │   │   ├── feature_dataloader.py
│   │   │   ├── openseg_dataloader.py
│   │   │   ├── openseg_extractor.py
│   ├── encoders
│   │   ├── image_encoder.py
│   ├── opennerf_config.py
│   ├── opennerf_datamanger.py
│   ├── opennerf_field.py
│   ├── opennerf_fieldheadnames.py
│   ├── opennerf_model.py
│   ├── opennerf_pipeline.py
│   ├── opennerf_renderers.py
├── pyproject.toml
```

## BibTeX
```
@inproceedings{engelmann2024opennerf,
  title={{OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views}},
  author={Engelmann, Francis and Manhardt, Fabian and Niemeyer, Michael and Tateno, Keisuke and Pollefeys, Marc and Tombari, Federico},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
