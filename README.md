# [CVPR 2026] CountGD++: Generalized Prompting for Open-World Counting
Niki Amini-Naieni & Andrew Zisserman

## [NOTE]: Full training and inference code will be released by June 3rd, 2026 (CVPR 2026 conference start date)

Official PyTorch implementation for CountGD++. Details can be found in the paper, [[Paper]](https://arxiv.org/abs/2512.23351) [[Project page]](https://github.com/niki-amini-naieni/CountGDPlusPlus/).

If you find this repository useful, please give it a star ⭐.

<img src=img/teaser.jpg width="100%"/>
<strong>New capabilities of CountGD++.</strong>
<em>(a) Counting with Positive & Negative Prompts:</em> The negative visual exemplar enables CountGD++ to differentiate between cells that have the same round shape as the object to count but are of a different appearance;  
<em>(b) Pseudo-Exemplars:</em> Pseudo-exemplars are automatically detected from text-only input and fed back to the model, improving the accuracy of the final count for objects, like unfamiliar fruits, that are challenging to identify given text alone.

## CountGD++ Architecture
<img src=img/inference-architecture.jpg width="100%"/>

## Contents
* [Demo](#demo)
* [Dataset Download](#dataset-download)
* [Reproduce Results From Paper](#reproduce-results-from-paper)
* [Training CountGD++](#training-countgd)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Demo
A Gradio graphical user interface demo has been created to allow users to test the model. The demo can be run on a remote GPU and accessed via a public link generated at runtime. A short video illustrating the demo workflow is included [here](https://drive.google.com/file/d/14cRslOiiEXqNrmOJsitIQliTqQIgAZ9C/view?usp=sharing). Note that pseudo-exemplars and adaptive cropping are not implemented in the demo. Please see the FSCD-147, PrACo, and ShanghaiTech test scripts to see how the pseudo-exemplars are implemented. Please see the FSCD-147 test script to see how adaptive cropping is implemented.

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

NOTE: In order to install detectron2 in step 4, you need to install the CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads. If multiple CUDA versions are installed, make sure you are using the right one. [This repository](https://github.com/phohenecker/switch-cuda) is quite useful for switching between CUDA versions.

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running and training CountGD++. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countgdplusplus python=3.10
conda activate countgdplusplus
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng # ensure to install required compilers
cd CountGDPlusPlus
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
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

* Download the pretrained CountGD++ model available [here](https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view?usp=sharing) (1.25 GB), and place it in the ```checkpoints``` directory or use ```gdown``` to download the weights.

  ```
  pip install gdown
  gdown --id 1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs
  ```

### 6. Run Demo

Run the command below to launch the demo. A video illustrating the demo workflow is provided [here](https://drive.google.com/file/d/14cRslOiiEXqNrmOJsitIQliTqQIgAZ9C/view?usp=sharing).

```
python app.py
```

## Dataset Download
By the end of this section, your directory should have the following structure:
```
CountGDPlusPlus\
  |data\
    |fscd147\
      |FSC147_384_V2\
        |annotations\
          annotation_FSC147_384.json
        |images_384_VarV2\
          1050.jpg
          ...
      |super-crops\
      |synthetic_exemplars\
      ...
    |blood_cell_detection\
      _annotations.coco.json
      external_exemplars.json
      image-100.png
      ...
    |omnicount_fruits_test\
    |ShanghaiTech\
      |part_A\
        |test_data\
        |train_data\
      |part_B\
  ...
```

### 1. Download FSCD-147
Download FSCD-147 from [here](https://drive.google.com/file/d/1m_v_hBwXH1NzcuUj_qa-ziKn-LYfUWA6/view?usp=sharing), and update [datasets_fscd147_test.json](config/datasets_fscd147_test.json) to point to the image folder you have downloaded.

For adaptive cropping, the image is cropped into smaller pieces, and AI-based super-resolution is applied to the crops before they are provided to the model. For the super-resolution, the image enhancer from the [Bubbi App](https://www.bubbi.app/tools/image-upscaler) was used. The image enhancer is not free, so the enhanced crops are provided [here](https://drive.google.com/file/d/1QyMfhPRp85La8URii8xL6wOoqcQpXhsD/view?usp=sharing). Please download and unzip the ```super-crops``` folder inside of the [fscd147](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/main/data/fscd147) folder for inference on FSCD-147.

For the text-only setting, synthetic exemplars were generated for the FSCD-147 test set using only the text. Please download and unzip the ```synthetic_exemplars``` folder [here](https://drive.google.com/file/d/1XyE77D0bxEKDsPYvpaeD6KyXnSLye1Qg/view?usp=sharing) inside of the [fscd147](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/main/data/fscd147) folder for inference on FSCD-147.

### 2. Download Blood Cell Detection
Download the images from the Blood Cell Detection repository [here](https://github.com/draaslan/blood-cell-detection-dataset) and place them in the [blood_cell_detection](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/main/data/blood_cell_detection) folder.

### 3. Download OmniCount (Fruits)
Download the images from the Omnicount-191 dataset [here](https://huggingface.co/datasets/cvssp/OmniCount-191/blob/main/OmniCount-191.zip), and place the images from the Fruits test set inside of the [omnicount_fruits_test](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/main/data/omnicount_fruits_test) folder.

### 4. Download ShanghaiTech Test
Download the ShanghaiTech dataset from [here](https://github.com/desenzhou/ShanghaiTechDataset). Create a new ```ShanghaiTech``` folder inside of the [data](https://github.com/niki-amini-naieni/CountGDPlusPlus/tree/main/data) folder, and place the part_A and part_B folders inside of the new ```ShanghaiTech``` folder.

## Reproduce Results From Paper

### 1. FSCD-147

* To test the text-only setting, run the following commands. Note that this setting uses both synthetic and pseudo-exemplars, obtained using only text.
  
  ```
  python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_plusplus.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_text_only.json" --fscd_gt_file data/fscd147/instances_test.json --num_exemplars 0 --use_synth_exemplars --crop --eval
  ```
  ```
  python test_fscd147.py --pred detections_test_text_only.json --gt data/fscd147/instances_test.json --split "test"
  ```
* To test the exemplar-only setting, run the following commands.
  
  ```
  python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_plusplus.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_exemplars_only.json" --fscd_gt_file data/fscd147/instances_test.json --no_text --num_exemplars 3 --crop --sam_tt_norm --eval
  ```
  ```
  python test_fscd147.py --pred detections_test_exemplars_only.json --gt data/fscd147/instances_test.json --split "test"
  ```
  
* To test the multi-modal setting, with both exemplars and text, run the following commands.

  ```
  python -u main_inference.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_plusplus.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_text_and_exemplars.json" --fscd_gt_file data/fscd147/instances_test.json --num_exemplars 3 --crop --sam_tt_norm --remove_bad_exemplar --eval
  ```
  ```
  python test_fscd147.py --pred detections_test_text_and_exemplars.json --gt data/fscd147/instances_test.json --split "test"
  ```

### 2. Blood Cell Detection

* To test the setting with positive text and 1 "positive internal exemplar" (a visual exemplar of the object to count from inside the input image), run the following commands:

  ```
  python test_dataset.py --dataset_folder data/blood_cell_detection --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --out_dir blood_cell_detection_output_pos_text_pos_int_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/blood_cell_detection/_annotations.coco.json --pred blood_cell_detection_output_pos_text_pos_int_exemp/coco_predictions.json
  ```

* To test the setting with positive text and 1 "positive external exemplar" (a visual exemplar of the object to count from one image applied across the dataset), run the following commands:

  ```
  python test_dataset.py --dataset_folder data/blood_cell_detection --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --use_ext_pos_exemp --out_dir blood_cell_detection_output_pos_text_pos_ext_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/blood_cell_detection/_annotations.coco.json --pred blood_cell_detection_output_pos_text_pos_ext_exemp/coco_predictions.json
  ```
* To test the setting with positive text, 1 "positive internal exemplar" (a visual exemplar of the object to count from inside the input image), negative text, and 1 "negative internal exemplar" (a visual exemplar of the object to *not* count from inside the input image), run the following command:

  ```
  python test_dataset.py --dataset_folder data/blood_cell_detection --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --neg_text --num_neg_exemp 1 --out_dir blood_cell_detection_output_pos_text_pos_int_exemp_neg_text_neg_int_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/blood_cell_detection/_annotations.coco.json --pred blood_cell_detection_output_pos_text_pos_int_exemp_neg_text_neg_int_exemp/coco_predictions.json
  ```

* To test the setting with positive text, 1 "positive external exemplar" (a visual exemplar of the object to count from one image applied across the dataset), negative text and 1 "negative external exemplar" (a visual exemplar of the object to *not* count from one image applied across the dataset), run the following command:

  ```
  python test_dataset.py --dataset_folder data/blood_cell_detection --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --use_ext_pos_exemp --neg_text --num_neg_exemp 1 --use_ext_neg_exemp --out_dir blood_cell_detection_output_pos_text_pos_ext_exemp_neg_text_neg_ext_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/blood_cell_detection/_annotations.coco.json --pred blood_cell_detection_output_pos_text_pos_ext_exemp_neg_text_neg_ext_exemp/coco_predictions.json
  ```
### 3. OmniCount (Fruits)

* To test the setting with positive text and 1 "positive internal exemplar" (a visual exemplar of the object to count from inside the input image), run the following commands:

  ```
  python test_dataset.py --dataset_folder data/omnicount_fruits_test --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --out_dir omnicount_fruits_test_output_pos_text_pos_int_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/omnicount_fruits_test/_annotations.coco.json --pred omnicount_fruits_test_output_pos_text_pos_int_exemp/coco_predictions.json
  ```

* To test the setting with positive text and 1 "positive external exemplar" (a visual exemplar of the object to count from one image applied across the dataset), run the following commands:

  ```
  python test_dataset.py --dataset_folder data/omnicount_fruits_test --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --use_ext_pos_exemp --out_dir omnicount_fruits_test_output_pos_text_pos_ext_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/omnicount_fruits_test/_annotations.coco.json --pred omnicount_fruits_test_output_pos_text_pos_ext_exemp/coco_predictions.json
  ```
* To test the setting with positive text, 1 "positive internal exemplar" (a visual exemplar of the object to count from inside the input image), negative text, and 1 "negative internal exemplar" (a visual exemplar of the object to *not* count from inside the input image) for each negative class, run the following command:

  ```
  python test_dataset.py --dataset_folder data/omnicount_fruits_test --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --neg_text --num_neg_exemp 1 --out_dir omnicount_fruits_test_output_pos_text_pos_int_exemp_neg_text_neg_int_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/omnicount_fruits_test/_annotations.coco.json --pred omnicount_fruits_test_output_pos_text_pos_int_exemp_neg_text_neg_int_exemp/coco_predictions.json
  ```

* To test the setting with positive text, 1 "positive external exemplar" (a visual exemplar of the object to count from one image applied across the dataset), negative text and 1 "negative external exemplar" (a visual exemplar of the object to *not* count from one image applied across the dataset) for each negative class, run the following command:

  ```
  python test_dataset.py --dataset_folder data/omnicount_fruits_test --pretrain_model_path checkpoints/countgd_plusplus.pth --pos_text --num_pos_exemp 1 --use_ext_pos_exemp --neg_text --num_neg_exemp 1 --use_ext_neg_exemp --out_dir omnicount_fruits_test_output_pos_text_pos_ext_exemp_neg_text_neg_ext_exemp
  ```
  ```
  python evaluate_coco_metrics.py --gt data/omnicount_fruits_test/_annotations.coco.json --pred omnicount_fruits_test_output_pos_text_pos_ext_exemp_neg_text_neg_ext_exemp/coco_predictions.json
  ```

### 4. ShanghaiTech Test
* To test CountGD++ on ShanghaiTech Part A Test given both text and pseudo-exemplars (exemplars automatically generated by CountGD++ using text only), run the following command:
  ```
  python test_shanghai_tech.py --pretrain_model_path checkpoints/countgd_plusplus.pth --image_folder data/ShanghaiTech/part_A/test_data/images/ --gt_folder data/ShanghaiTech/part_A/test_data/ground-truth/
  ```
* To test CountGD++ on ShanghaiTech Part B Test given both text and pseudo-exemplars (exemplars automatically generated by CountGD++ using text only), run the following command:
  ```
  python test_shanghai_tech.py --pretrain_model_path checkpoints/countgd_plusplus.pth --image_folder data/ShanghaiTech/part_B/test_data/images/ --gt_folder data/ShanghaiTech/part_B/test_data/ground-truth/
  ```
### 5. CARPK
To test CountGD++ on the CARPK test set given both text and pseudo-exemplars (exemplars automatically generated by CountGD++ using text only), run the following command:
```
python test_carpk.py 
```

### 5. VideoCount (Crystals)
Please see the instructions [here](https://github.com/niki-amini-naieni/CountVid#6-science-count-crystals-pseudo-exemplars) in the CountVid repository for testing the pseudo-exemplars capability on videos in the VideoCount (Crystals) dataset.

### 6. PrACo

### 7. PairTally


## Training CountGD++

## Citation
Please cite our related papers if you build off of our work.
```
@InProceedings{AminiNaieni26b,
  title={CountGD++: Generalized Prompting for Open-World Counting},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@InProceedings{AminiNaieni26a,
  title={Open-World Object Counting in Videos},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle = {Association for Advancement of Artificial Intelligence Conference (AAAI)},
  year={2026}
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
