# KG-TrajGen

The official implementation of **KG-TrajGen** .
KG-TrajGen explicitly models local and global spatial dependencies by constructing an RKG 
to learn informative road segment representations; it further characterizes the interaction 
between roads and the urban environment using E-RKG, and then effectively integrates environmental 
information into the trajectory generation process through an MLP fusion module.

## Installation

- Environment
    - Tested OS: Linux
    - Python >= 3.9
    - PyTorch == 1.11.0

- Dependencies
    - Install PyTorch 1.11.0 with the correct CUDA version.
    - Install dgl 1.1.0 with the correct CUDA version.
    - Execute ``pip install -r requirements.txt`` command to install all of the Python modules and packages used in this project.
  
## Pre-Training
- Meta Data Collection
  POI & Functional Area & Administrative Borough & Road Meta data is from [OpenStreetMap](https://www.openstreetmap.org/)
  
- Preprocess Mete Data
  - `cd ./kg_pretrain/UrbanKG_data`
  - `python preprocess_meta_data.py`
  
- Construct RKG
  - `python construct_base_KG.py`
  
- Construct E-RKG
  - `python construct_augmented_KG.py`
  
- Pre-process E-RKG
  - `cd ./kg_pretrain/UrbanKG_data`
  - `python datasets/process.py`
  
- Run KG Embedding Model
  - `python run.py --model GIE --multi_c`
  - `--dataset` sets the dataset: `bj or porto`

The implemention is based on *[UUKG](https://github.com/usail-hkust/UUKG)*.
## Running

Trajectory Data is from the opensource dataset of [TS-TrajGen](https://github.com/WenMellors/TS-TrajGen/tree/master).

- Training Phase
  - `python train.py`
  - `--data` sets the dataset
  - `--datapath` refers to the path of each dataset
  - `--out_dir` is the file directory to save the trained model
  
- Inference Phase
  - `python sample.py`
  - `--data` sets the dataset
  - `--datapath` refers to the path of each dataset
  - `--out_dir` is the file directory to save the trained model and simulated trajectories
  
- Evaluation
  - `python my_evaluations.py`
  - `--datasets` sets the dataset
  
## Note

The implemention is based on *[nanoGPT](https://github.com/karpathy/nanoGPT)* & *[STEGA](https://github.com/Star607/STEGA)*.
