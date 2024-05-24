# GMMFormer v2: An Uncertainty-aware Framework for Partially Relevant Video Retrieval

This repository is the official PyTorch implementation of our paper [GMMFormer v2: An Uncertainty-aware Framework for Partially Relevant Video Retrieval](https://arxiv.org/pdf/2405.13824).


## Catalogue <br> 
* [1. Getting Started](#getting-started)
* [2. Run](#run)
* [3. Trained Models](#trained-models)
* [4. Results](#results)
* [5. Citation](#citation)



## Getting Started

1\. Clone this repository:
```
git clone https://github.com/huangmozhi9527/GMMFormer_v2.git
cd GMMFormer_v2
```

2\. Create a conda environment and install the dependencies:
```
conda create -n prvr python=3.9
conda activate prvr
conda install pytorch==1.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

3\. Download Datasets: All features of TVR, ActivityNet Captions and Charades-STA are kindly provided by the authors of [MS-SL].


4\. Set root and data_root in config files (*e.g.*, ./Configs/tvr.py).

## Run

To train GMMFormer_v2 on TVR:
```
cd src
python main.py -d tvr --gpu 0
```

To train GMMFormer_v2 on ActivityNet Captions:
```
cd src
python main.py -d act --gpu 0
```

To train GMMFormer_v2 on Charades-STA:
```
cd src
python main.py -d cha --gpu 0
```



## Trained Models

We provide trained GMMFormer_v2 checkpoints. You can download them from Baiduyun disk.

| *Dataset* | *ckpt* |
| ---- | ---- |
| TVR | [Baidu disk](https://pan.baidu.com/s/1GbHBvnr5Y7Tz43HU4K2p2w?pwd=9527) |
| ActivityNet Captions | [Baidu disk](https://pan.baidu.com/s/1nmgfyjg4SgeC9NM2kg02wg?pwd=9527) |
| Charades-STA | [Baidu disk](https://pan.baidu.com/s/1-_SBrQ1Tla-Rut-fdtnqCw?pwd=9527) |

## Results

### Quantitative Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
| ---- | ---- | ---- | ---- | ---- | ---- |
| TVR | 16.2 | 37.6 | 48.8 | 86.4 | 189.1 |
| ActivityNet Captions | 8.9 | 27.1 | 40.2 | 78.7 | 154.9 |
| Charades-STA | 2.5 | 8.6 | 13.9 | 53.2 | 78.2 |


[MS-SL]:https://github.com/HuiGuanLab/ms-sl



