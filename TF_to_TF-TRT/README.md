# Converting to Tensorflow saved_model and TensorRT
You can convert trained checkpoints to saved_model type for using in your scripts.

## Usage

### Step 1. Clone repository
```
git clone https://github.com/bigalex95/pix2pix_TecoGAN
cd ./TF_to_TF-TRT
```
### Step 2. Create virtualenv and install necessary libraries.
```
# create virtualenv
python -m venv venv_name
# change environment to virtualenv
## Windows 
### 1. cmd.exe
./venv_name\Scripts\activate
### 2. PowerShell
./venv_name/Scripts/activate.ps1
## Linux bash
./venv_name/bin/activate
# install packages
pip install -r requirements.txt
```
### Step 3. Install TensorRT.

How to install Nvidia TensorRT need check this [link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

### Step 4. Run [checkpoint2saved_model.py](./checkpoint2saved_model.py) script for creating saved_model.
### Step 5. Run [TF-TRT.py](./TF-TRT.py) scipt for creating optimized model from saved_model.