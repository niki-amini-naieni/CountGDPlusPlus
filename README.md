# CountGD++: Generalized Prompting for Open-World Counting
## [NOTE]: Code and models will be released within the next few days (by Jan 1, 2026)
The release will include a single-image inference demo (with positive/negative prompts), training, and evaluation code, along with pretrained checkpoints and dataset instructions.

Niki Amini-Naieni & Andrew Zisserman

Official PyTorch implementation for CountGD++. Details can be found in the paper, [[Paper]]() [[Project page]](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

If you find this repository useful, please give it a star ⭐.

<img src=img/teaser.jpg width="100%"/>
<strong>New capabilities of CountGD+.</strong>
<em>(a) Counting with Positive & Negative Prompts:</em> The negative visual exemplar enables CountGD++ to differentiate between cells that have the same round shape as the object to count but are of a different appearance;  
<em>(b) Pseudo-Exemplars:</em> Pseudo-exemplars are automatically detected from text-only input and fed back to the model, improving the accuracy of the final count for objects, like unfamiliar fruits, that are challenging to identify given text alone.

## CountGD++ Architecture
<img src=img/inference-architecture.jpg width="100%"/>

## Contents
* [Demo](#demo)
* [Dataset Download](#dataset-download)
* [Reproduce Results From Paper](#reproduce-results-from-paper)
* [Training CountGD++](#training-countgd++)
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

NOTE: In order to install detectron2 in step 4, you needed to install CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running and training CountGD++. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

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

* Download the pretrained CountGD++ model available [here](), and place it in the ```checkpoints``` directory Or use ```gdown``` to download the weights.

  ```
  pip install gdown
  gdown --id [tbd]
  ```

### 6. Run Demo

## Dataset Download

## Reproduce Results From Paper

## Training CountGD++

## Citation
Please cite our related papers if you build off of our work.
```
@article{AminiNaieni25,
  title={CountGD++: Generalized Prompting for Open-World Counting},
  author={Amini-Naieni, N. and Zisserman, A.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}

@InProceedings{AminiNaieni24,
  title = {CountGD: Multi-Modal Open-World Counting},
  author = {Amini-Naieni, N. and Han, T. and Zisserman, A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
```

## Acknowledgements
The authors would like to thank Dr Christian Schroeder de Witt (Oxford Witt Lab, OWL) for his helpful feedback and insights on the paper figures and Gia Khanh Nguyen, Yifeng Huang, and Professor Minh Hoai for their help with the PairTally Benchmark. This research is funded by an AWS Studentship, the Reuben Foundation, a Qualcomm Innovation Fellowship (mentors: Dr Farhad Zanjani and Dr Davide Abati), the AIMS CDT program at the University of Oxford, EPSRC Programme Grant VisualAI EP/T028572/1, and a Royal Society Research Professorship RSRP\R\241003.
