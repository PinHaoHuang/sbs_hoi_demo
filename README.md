# sbs_hoi_demo

## Install third_party 

### 4D-Humans

Navigate to https://github.com/shubham-goel/4D-Humans. Follow their instructions to install the environment.
This should create a conda environment named 
`
4D-humans
`
.
In addition, to use fastSAM, run
`
pip install ultralytics
`

### mmpose

Navigate to https://github.com/open-mmlab/mmpose. Follow their instructions to install the environment.
This should create a conda environment named
`
openmmlab
`

## Create your own environment

```
conda create -n sbs_hoi_demo python=3.10
conda activate sbs_hoi_demo
```

Install pytorch:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install other required package via pip:
``` 
streamlit
opencv-python
scikit-learn
open3d
videoio
zmq
numpy==1.26.4
```
