# CountGD++: Generalized Prompting for Open-World Counting

Niki Amini-Naieni & Andrew Zisserman

Official PyTorch implementation for CountGD++. Details can be found in the paper, [[Paper]]() [[Project page]](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

If you find this repository useful, please give it a star ⭐.

<img src=img/teaser.jpg width="100%"/>
<strong>New capabilities of COUNTGD++.</strong>
<em>(a) Counting with Positive & Negative Prompts:</em> The negative visual exemplar enables CountGD++ to differentiate between cells that have the same round shape as the object to count but are of a different appearance;  
<em>(b) Pseudo-Exemplars:</em> Pseudo-exemplars are automatically detected from text-only input and fed back to the model, improving the accuracy of the final count for objects, like unfamiliar fruits, that are challenging to identify given text alone.

## CountGD++ Architecture
<img src=img/inference-architecture.jpg width="100%"/>

## Contents
* [Demo](#demo)
* [Dataset Download](#dataset-download)
* [Reproduce Results From Paper](#reproduce-results-from-paper)
* [Training CountGD++](#training-countgd-box)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Demo

### 1. Clone Repository

```
git clone git@github.com:niki-amini-naieni/CountGDPlusPlus.git
```

### 2. Install GCC 

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```
sudo apt update
sudo apt install build-essential
sudo apt install gcc-11 g++-11
```

### 3. Install CUDA Toolkit:

NOTE: In order to install detectron2 in step 4, you needed to tnstall CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running CountVid and training CountGD++. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countgdplusplus python=3.10
conda activate countgdplusplus
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng # ensure to install required compilers
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 5. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```CountGDPlusPlus``` repository.

  ```
  mkdir checkpoints
  ```

* Execute the following command.

  ```
  python download_bert.py
  ```

* Download the pretrained CountGD++ model available [here](https://drive.google.com/file/d/1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD/view?usp=sharing), and place it in the ```checkpoints``` directory Or use ```gdown``` to download the weights.

  ```
  pip install gdown
  gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/
  ```

### 6. Run Demo
* Run the following command.
```
python count_in_videos.py --video_dir demo --input_text "penguin" --sam_checkpoint checkpoints/sam2.1_hiera_large.pt --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --obj_batch_size 30 --img_batch_size 10 --downsample_factor 1 --pretrain_model_path checkpoints/countgd_box.pth --temp_dir ./demo_temp --output_dir ./demo_output --save_final_video --save_countgd_video
```
* Visualize the output.
You should see the following videos saved to the ```demo_output``` folder once the demo has finished running:

  ```final-video.mp4```
  <p align="center">
      <img src="./img/final-video-demo.gif" alt="final output video" width="100%"/>
  </p>

  ```countgd-video.avi```
  <p align="center">
      <img src="./img/countgd-video-demo.gif" alt="timelapse boxes from CountGD-Box" width="100%"/>
  </p>

