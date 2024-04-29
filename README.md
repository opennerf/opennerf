# OpenNeRF: OpenSet 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views

Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, Federico Tombari

--- ICLR 2024 ---

![OpenNeRF Teaser](https://opennerf.github.io/static/images/teaser.png)

### Setup

#### Install NerfStudio

After installing conda (see [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)), setup the conda environment:

```
conda create --name opennerf -y python=3.8
conda activate opennerf
python -m pip install --upgrade pip
```

### Install cuda, torch, etc.
```
conda install nvidia/label/cuda-12.1.1::cuda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
### Install OpenNeRF
```
git clone https://github.com/opennerf/opennerf
cd opennerf
python -m pip install -e .
ns-install-cli
```

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

## Running OpenNeRF
This repository creates a new Nerfstudio method named "opennerf". To train with it, run the command:
```
ns-train opennerf --data [PATH]
```
See `.vscode/launch.json` for specific examples.


## BibTeX
```
@inproceedings{engelmann2024opennerf,
  title={{OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views}},
  author={Engelmann, Francis and Manhardt, Fabian and Niemeyer, Michael and Tateno, Keisuke and Pollefeys, Marc and Tombari, Federico},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
