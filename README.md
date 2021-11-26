# Domain Adaptive Video Segmentation via Temporal Consistency Regularization

### Updates
- *08/2021*: check out our domain adaptation for sematic segmentation paper [RDA: Robust Domain Adaptation via Fourier Adversarial Attacking](https://arxiv.org/abs/2106.02874) (accepted to ICCV 2021). This paper presents RDA, a robust domain adaptation technique that introduces adversarial attacking to mitigate overfitting in UDA. [Code avaliable](https://github.com/jxhuang0508/RDA).
- *06/2021*: check out our domain adaptation for panoptic segmentation paper [Cross-View Regularization for Domain Adaptive Panoptic Segmentation](https://arxiv.org/abs/2103.02584) (accepted to CVPR 2021). We design a domain adaptive panoptic segmentation network that exploits inter-style consistency and inter-task regularization for optimal domain adaptation in panoptic segmentation.[Code avaliable](https://github.com/jxhuang0508/FSDR).
- *06/2021*: check out our domain generalization paper [FSDR: Frequency Space Domain Randomization for Domain Generalization](https://arxiv.org/abs/2103.02370) (accepted to CVPR 2021). Inspired by the idea of JPEG that converts spatial images into multiple frequency components (FCs), we propose Frequency Space Domain Randomization (FSDR) that randomizes images in frequency space by keeping domain-invariant FCs and randomizing domain-variant FCs only. [Code avaliable](https://github.com/jxhuang0508/CVRN).
- *06/2021*: check out our domain adapation for sematic segmentation paper [Scale variance minimization for unsupervised domain adaptation in image segmentation](https://www.researchgate.net/publication/347421562_Scale_variance_minimization_for_unsupervised_domain_adaptation_in_image_segmentation)  (accepted to Pattern Recognition 2021). We design a scale variance minimization (SVMin) method by enforcing the intra-image semantic structure consistency in the target domain. [Code avaliable](https://github.com/Dayan-Guan/SVMin).
- *06/2021*: check out our domain adapation for object detection paper [Uncertainty-Aware Unsupervised Domain Adaptation in Object Detection](https://arxiv.org/abs/2103.00236) (accepted to IEEE TMM 2021). We design a uncertainty-aware domain adaptation network (UaDAN) that introduces conditional adversarial learning to align well-aligned and poorly-aligned samples separately in different manners. [Code avaliable](https://github.com/Dayan-Guan/UaDAN).

### Paper
![](./teaser.jpg)

[Domain Adaptive Video Segmentation via Temporal Consistency Regularization](https://arxiv.org/abs/2107.11004) [Dayan Guan](https://scholar.google.com/citations?user=9jp9QAsAAAAJ&hl=en), [Jiaxing Huang](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao),  [Xiao Aoran](https://scholar.google.com/citations?user=yGKsEpAAAAAJ&hl=en), [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)  
 School of Computer Science and Engineering, Nanyang Technological University, Singapore  
 International Conference on Computer Vision, 2021.
 
If you find this code useful for your research, please cite our [paper](https://arxiv.org/abs/2107.11004):

```
@inproceedings{guan2021domain,
  title={Domain adaptive video segmentation via temporal consistency regularization},
  author={Guan, Dayan and Huang, Jiaxing and Xiao, Aoran and Lu, Shijian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8053--8064},
  year={2021}
}
```

### Abstract
Video semantic segmentation is an essential task for the analysis and understanding of videos. Recent efforts largely focus on supervised video segmentation by learning from fully annotated data, but the learnt models often experience clear performance drop while applied to videos of a different domain. This paper presents DA-VSN, a domain adaptive video segmentation network that addresses domain gaps in videos by temporal consistency regularization (TCR) for consecutive frames of target-domain videos. DA-VSN consists of two novel and complementary designs. The first is cross-domain TCR that guides the prediction of target frames to have similar temporal consistency as that of source frames (learnt from annotated source data) via adversarial learning. The second is intra-domain TCR that guides unconfident predictions of target frames to have similar temporal consistency as confident predictions of target frames. Extensive experiments demonstrate the superiority of our proposed domain adaptive video segmentation network which outperforms multiple baselines consistently by large margins.

### Installation
1. Conda enviroment:
```bash
conda create -n DA-VSN python=3.6
conda activate DA-VSN
conda install -c menpo opencv
pip install torch==1.2.0 torchvision==0.4.0
```

2. Clone the [ADVENT](https://github.com/valeoai/ADVENT):
```bash
git clone https://github.com/valeoai/ADVENT.git
pip install -e ./ADVENT
```

3. Clone the repo:
```bash
git clone https://github.com/Dayan-Guan/DA-VSN.git
pip install -e ./DA-VSN
```

### Preparation 
1. Dataset:
* [Cityscapes-Seq](https://www.cityscapes-dataset.com/)
```bash
DA-VSN/data/Cityscapes/                       % Cityscapes dataset root
DA-VSN/data/Cityscapes/leftImg8bit_sequence   % leftImg8bit_sequence_trainvaltest
DA-VSN/data/Cityscapes/gtFine                 % gtFine_trainvaltest
```

* [VIPER](https://playing-for-benchmarks.org/download/): 
```bash
DA-VSN/data/Viper/                            % VIPER dataset root
DA-VSN/data/Viper/train/img                   % Modality: Images; Frames: *[0-9]; Sequences: 00-77; Format: jpg
DA-VSN/data/Viper/train/cls                   % Modality: Semantic class labels; Frames: *0; Sequences: 00-77; Format: png
```

* [SYNTHIA-Seq](http://synthia-dataset.cvc.uab.cat/SYNTHIA_SEQS/SYNTHIA-SEQS-04-DAWN.rar) 
```bash
DA-VSN/data/SynthiaSeq/                      % SYNTHIA-Seq dataset root
DA-VSN/data/SynthiaSeq/SEQS-04-DAWN          % SYNTHIA-SEQS-04-DAWN
```

2. Pre-trained models:
Download [pre-trained models](https://github.com/Dayan-Guan/DA-VSN/releases/tag/Latest) and put in ```DA-VSN/pretrained_models```

### Optical Flow Estimation
* For quick preparation: Download the optical flow estimated from Cityscapes-Seq validation set [here](https://github.com/Dayan-Guan/DA-VSN/releases/tag/Latest) and unzip in ```DA-VSN/data```

1. Clone the [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch):
```bash
git clone https://github.com/NVIDIA/flownet2-pytorch.git
```

2. Download [pre-trained FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) and put in ```flownet2-pytorch/pretrained_models```

```bash
DA-VSN/data/Cityscapes_val_optical_flow_scale512/  % unzip Cityscapes_val_optical_flow_scale512.zip
```

3. Use the [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) to estimate optical flow

### Evaluation on Pretrained Models
* VIPER → Cityscapes-Seq: 
```bash
cd DA-VSN/davsn/scripts
python test.py --cfg configs/davsn_viper2city_pretrained.yml
```

* SYNTHIA-Seq → Cityscapes-Seq: 
```bash
python test.py --cfg configs/davsn_syn2city_pretrained.yml
```

### Training and Testing
* VIPER → Cityscapes-Seq: 
```bash
cd DA-VSN/davsn/scripts
python train.py --cfg configs/davsn_viper2city.yml
python test.py --cfg configs/davsn_viper2city.yml
```

* SYNTHIA-Seq → Cityscapes-Seq: 
```bash
python train.py --cfg configs/davsn_syn2city.yml
python test.py --cfg configs/davsn_syn2city.yml
```

## Acknowledgements
This codebase is heavily borrowed from [ADVENT](https://github.com/valeoai/ADVENT) and [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch).

## Contact
If you have any questions, please contact: dayan.guan@ntu.edu.sg
