<span align="center">
<h1> OpenNeRF: OpenSet 3D Neural Scene Segmentation<br>with Pixel-Wise Features and Rendered Novel Views</h1>

<a href="https://francisengelmann.github.io">Francis Engelmann</a>,
<a href="https://scholar.google.de/citations?user=bERItx8AAAAJ">Fabian Manhardt</a>,
<a href="https://m-niemeyer.github.io">Michael Niemeyer</a>,
<a href="https://scholar.google.com/citations?user=ml3laqEAAAAJ">Keisuke Tateno</a>,
<a href="https://inf.ethz.ch/people/person-detail.pollefeys.html">Marc Pollefeys</a>,
<a href="https://federicotombari.github.io">Federico Tombari</a>

<h3>ICLR 2024</h3>

<a href="https://arxiv.org/abs/2404.03650">Paper</a> |
<a href="http://opennerf.github.io">Project Page</a> |
<a href="http://opennerf.github.io/demo.html">Demo</a>

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
pip install --upgrade --force-reinstall pillow
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
unzip Replica.zip && mv Replica replica
cd ..
python datasets/preprocess.py --dataset-name=replica
```

## SceneFun3D Dataset

Set up the SceneFun3D repository to download the dataset:
```
git clone https://github.com/SceneFun3D/scenefun3d.git
cd scenefun3d
conda create --name scenefun3d python=3.8
conda activate scenefun3d
pip install -r requirements.txt
```

Download the dataset (make sure to put the correct download_dir):
```
python -m data_downloader.data_asset_download --split test_set --download_dir /path/to/large_disk/data --download_only_one_video_sequence --dataset_assets arkit_mesh lowres_wide	lowres_depth
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

## Replica: Semantic Predictions and Evaluation

Adapt the global variables `CONDA_DIR` and `PREFIX` in `train_eval_replica_semantics.py`. Then run it:

```
python scripts/train_eval_replica_semantics.py
```

This version of the code corresponds to entry (2) "Render & Project" in Table 2 of the paper and produces the following 3D semantic segmentation scores on the Replica dataset:

<table>
  <tr>
    <td></td>
    <td colspan=2 align="center"><b>All</b></td>
    <td colspan=2 align="center"><b>Head</b></td>
    <td colspan=2 align="center"><b>Common</b></td>
    <td colspan=2 align="center"><b>Tail</b></td>
  </tr>
  <tr>
    <td></td>
    <td align="center">mIoU</td><td align="center">mAcc</td>
    <td align="center">mIoU</td><td align="center">mAcc</td>
    <td align="center">mIoU</td><td align="center">mAcc</td>
    <td align="center">mIoU</td><td align="center">mAcc</td>
  </tr>
  <tr>
    <td>Run 0</td>
    <td align="center">18.74%</td><td align="center">31.87%</td>
    <td align="center">30.26%</td><td align="center">43.89%</td>
    <td align="center">20.07%</td><td align="center">33.52%</td>
    <td align="center">5.88%</td><td align="center">18.19%</td>
  </tr>
  <tr>
    <td>Run 1</td>
    <td align="center">19.68%</td><td align="center">32.68%</td>
    <td align="center">30.76%</td><td align="center">44.70%</td>
    <td align="center">20.84%</td><td align="center">34.16%</td>
    <td align="center">7.43%</td><td align="center">19.17%</td>
  </tr>
  <tr>
    <td>Run 2</td>
    <td align="center">18.80%</td><td align="center">31.72%</td>
    <td align="center">30.36%</td><td align="center">44.02%</td>
    <td align="center">19.63%</td><td align="center">32.71%</td>
    <td align="center">6.41%</td><td align="center">18.43%</td>
  </tr>
</table>


## Coordinate Frames

![Coordindate_systems](https://opennerf.github.io/static/images/coordinate_systems.jpg)

## BibTeX
If you find our code or paper useful, please cite:
```bibtex
@inproceedings{engelmann2024opennerf,
  title     = {{OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views}},
  author    = {Engelmann, Francis and Manhardt, Fabian and Niemeyer, Michael and Tateno, Keisuke and Pollefeys, Marc and Tombari, Federico},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
```
