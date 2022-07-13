# Personalizing Federated Medical Image Segmentation via Local Calibration

## Introduction

This is an official release of the paper **Personalizing Federated Medical Image Segmentation via Local Calibration**, including the network implementation and the training scripts.

> [**Personalizing Federated Medical Image Segmentation via Local Calibration**](https://arxiv.org/abs/2207.04655),   <br/>
> **Jiacheng Wang**, Yueming Jin, Liansheng Wang <br/>
> In: European Conference on Computer Vision (ECCV), 2022  <br/>
> [[arXiv](https://arxiv.org/abs/2207.04655)][[Bibetex](https://github.com/jcwang123、FedLC#citation)]

<div align="center" border=> <img src=frame.png width="700" > </div>


## News
- **[7/12 2022] We have released the pre-print manuscript.**
- **[7/11 2022] We have released the pre-trained weights on the polyp segmentation.**
- **[7/4 2022] We have released the pre-processing scripts.**
- **[7/4 2022] We have created this repo.**

## Code List

- [x] Network
- [x] Pre-processing
- [ ] Training Codes
- [x] Pretrained Weights

For more details or any questions, please feel easy to contact us by email (jiachengw@stu.xmu.edu.cn).

## Usage

### Dataset
In this paper, we perform the experiments using three imaging modalities, including the polyp images, fundus images, and prostate MR images. They could be downloaded from the public websites, or copied from [FedDG](https://github.com/liuquande/FedDG-ELCFS) and [PraNet](https://github.com/DengPingFan/PraNet).

### Pre-processing
After downloading the data resources, please run the file `utils/prepare_dataset.py`. Note that the file directory should be replaced with yours.

### Training 
TODO

### Testing
Please download the pre-trained weights from Baidu Disk (https://pan.baidu.com/s/10HkQ90xeFcHMaNgfIyT0iw, a1sm) and put them in the project directory.

Rename the directory as `logs/{dataset}/{exp_name}/model/`.

Run the test script `test.py`.

### Result
The test **IoU** scores and **ASSD** scores on the Polyp dataset are:

<div align="center" border=> <img src=result.png width="700" > </div>

## Citation
If you find LC-Fed useful in your research, please consider citing:
```
@misc{wang2022personalizing,
      title={Personalizing Federated Medical Image Segmentation via Local Calibration}, 
      author={Jiacheng Wang and Yueming Jin and Liansheng Wang},
      year={2022},
      eprint={2207.04655},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
