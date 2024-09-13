# EHNet Efficient Hybrid Network with Dual Attention for Image Deblurring

Quoc-Thien Ho, Minh-Thien Duong, Seongsoo Lee and Min-Cheol Hong
> Abstract: The motion of an object or camera platform makes the acquired image blurred. This degradation
is a major reason to obtain a poor-quality image from an imaging sensor. Therefore, developing an efficient
deep-learning-based image processing method to remove the blur artifact is desirable. Deep learning has
recently demonstrated significant efficacy in image deblurring, primarily through convolutional neural
networks (CNNs) and Transformers. However, the limited receptive fields of. CNNs restrict their ability
to capture long-range structural dependencies. In contrast, Transformers excel at modeling these dependencies,
but they are computationally expensive for high-resolution inputs and lack the appropriate inductive bias.
To overcome these challenges, we propose the Efficient Hybrid Network (EHNet) that employs CNN encoders for
local feature extraction and Transformer decoders with a dual-attention module to capture spatial and
channel-wise dependencies. This synergy facilitates the acquisition of rich contextual information for
high-quality image deblurring. Additionally, we introduce the Simple Feature Embedding Module (SFEM) to
replace the pointwise and depthwise convolutions to generate simplified embedding features in the self-attention
mechanism. This innovation substantially reduces computational complexity and memory usage while maintaining
overall performance. Finally, through comprehensive experiments, our compact model yields promising quantitative
and qualitative results for image deblurring on various benchmark datasets.

## Installation 
This project is built with Python 3.12, Pytorch 2.3, CUDA 12.4, Cudnn-cuda-12, anaconda.

For installing, follow these instructions:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install natsort opencv-python einops ptflops lmdb tqdm scikit-image warmup_scheduler
```
## Dataset 
- Download deblur dataset: [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro.html), [HIDE dataset](https://github.com/joanshen0508/HA_deblur?tab=readme-ov-file), [RealBlur](https://cg.postech.ac.kr/research/realblur/).

- Preprocess data folder. The data folder should be like the format:
  
- GOPRO / HIDE / RealBlur_J / RealBlur_R


├─ test

│ ├─ input    % GoPro 1111 image pairs, HIDE 2025 image pairs, RealBLur_J 980 image pairs, RealBlur_R 980 image pairs.

│ │ ├─ xxxx.png

│ │ ├─ ......

│ │

│ ├─ target

│ │ ├─ xxxx.png

│ │ ├─ ......

│
├─ train   

│ ├─ ...... (same as test)

-  Processed test data:
[GoPro dataset](https://drive.google.com/drive/folders/1UJBDkoIngLHeHAPraB80CDZT1xnWSg-S?usp=drive_link),
[HIDE dataset](https://drive.google.com/drive/folders/1UO6Inn6mNur0CRjjyr9QxAtpR9szPbbM?usp=drive_link),
[RealBlur_J](https://drive.google.com/drive/folders/1-HlxxzWrrxCytQDOM6bZPNCdUMOfIkWe?usp=drive_link),
[RealBlur_R](https://drive.google.com/drive/folders/1oLF5BwrBa99U4y5KqzpbPX6TYjkNc0iw?usp=drive_link).

## Training
To train EHNet, run the command line as follows:
```
python train.py --target_dir path_to_target --input_dir path_to_input --model_save_dir path_to_save_model
```
## Testing
```
python test.py --target_dir path_to_target --input_dir path_to_input --weights path_to_pretrained_weights --output_dir path_to_save_image 
```
## Evaluation
- For evaluation GoPro / HIDE dataset. Compare the results output with target images of the test set.
```
evaluate_GOPRO.m
```
 For evaluation Realblur_J / RealBlur_R dataset.  Compare the results output with target images of the test set.
```
evaluate_realblur.py
```
## Reference Code:

https://github.com/chosj95/MIMO-UNet/tree/main
https://github.com/INVOKERer/DeepRFT
https://github.com/swz30/Restormer

