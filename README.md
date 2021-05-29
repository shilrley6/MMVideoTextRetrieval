# MMVideoTextRetrieval
MMVideoTextRetrieval is an open source video-text retrieval toolbox based on PyTorch.



## Introduction 

This repository provides different video text retrieval methods.

### Major Features 

* **Modular design**

  We decompose the video-text retrieval framework into different components which can be easily used any combination.

* **Support for various datasets and features**

  The toolbox supports multiple datasets, such as MSRVTT, ActivityNet, LSMDC. Besides, various extracted features are provided.

* **Support for multiple video text retrieval frameworks**

  MMVideoTextRetrieval  implements popular frameworks for video text retrieval, such as MMT, etc. More frameworks will be added later.

* **Visual demo**

  We provide the demo to visualize the results of video text retrieval models.

  

## Demo
  We provide a way to produce text-to-video retrieval in real-world applications. Before retrieval, the multi-model features of videos should be extracted and stored. The searched text is defined in the "main_train" function in demo.py, and the config "--sentence" should be used to activate the retrieval process. The outputs of the retrieval are the name of video feature files of the top 10 similar videos.


## Benchmark

<table>
   <tr>
      <td>Model</td>
      <td>Dataset</td>
      <td>Video Feature</td>
      <td>Text Feature </td>
      <td>Pretrained</td>
      <td>Text-to-Video Retrieval</td>
      <td></td>
      <td></td>
      <td>Video-to-Text Retrieval</td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
      <td>R@1</td>
      <td>R@5</td>
      <td>R@10</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>MSTVTT-1kA</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>no</td>
      <td>24.6</td>
      <td>54</td>
      <td>67.1</td>
      <td>24.4</td>
      <td>56</td>
      <td>67.8</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>ActivityNet</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>no</td>
      <td>22.7</td>
      <td>54.2</td>
      <td>93.2</td>
      <td>22.9</td>
      <td>54.8</td>
      <td>93.1</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>LSMDC</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>no</td>
      <td>13.2</td>
      <td>29.2</td>
      <td>38.8</td>
      <td>12.1</td>
      <td>29.3</td>
      <td>37.9</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>MSTVTT-1kA&B</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>HowTo100M</td>
      <td>26.6</td>
      <td>57.1</td>
      <td>69.6</td>
      <td>27</td>
      <td>57.5</td>
      <td>69.7</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>ActivityNet</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>HowTo100M</td>
      <td>28.7</td>
      <td>61.4</td>
      <td>94.5</td>
      <td>28.9</td>
      <td>61.1</td>
      <td>94.3</td>
   </tr>
   <tr>
      <td>MMT</td>
      <td>LSMDC</td>
      <td>S3D</td>
      <td>Bert</td>
      <td>HowTo100M</td>
      <td>12.9</td>
      <td>29.9</td>
      <td>40.1</td>
      <td>12.3</td>
      <td>28.6</td>
      <td>38.9</td>
   </tr>
   <tr>
      <td>HGR</td>
      <td>MSTVTT-Full</td>
      <td>Resnet152</td>
      <td>Word2Vec</td>
      <td>no</td>
      <td>9.2</td>
      <td>26.2</td>
      <td>36.5</td>
      <td>15</td>
      <td>36.7</td>
      <td>48.8</td>
   </tr>
</table>

(All the results are excerpted from the original paper and will be replaced by the results of pre-trained models later.)



## Model Zoo

supported methods for Video Text retrieval.

- [x] MMT (ECCV'2020)

- [x] MMT-modified (ICMEW'2021)

- [ ] HGR (CVPR'2020)



## Dataset

supported datasets.

<details open>
<summary>(click to collapse)</summary>

* MSR-VTT

  * [x] [raw dataset](http://ms-multimedia-challenge.com/2017/dataset)
  * [x] [multi-modal features](http://thoth.inrialpes.fr/research/video-features/)
  * [x] [Resnet152 video features](https://github.com/cshizhe/hgr_v2t)

* ActivityNet Captions

  - [x] [raw dataset](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

  - [x] [multi-modal features](http://thoth.inrialpes.fr/research/video-features/)

* LSMDC

  - [x] [raw dataset](https://sites.google.com/site/describingmovies/home)

  - [x] [multi-modal features](http://thoth.inrialpes.fr/research/video-features/)

* TGIF 
  - [x] [raw dataset](http://raingo.github.io/TGIF-Release/)
  - [x] [Resnet152 video features](https://github.com/cshizhe/hgr_v2t)

* VATEX
  - [x] [raw dataset](https://eric-xw.github.io/vatex-website/download.html)
  - [x] [I3D video features](https://github.com/cshizhe/hgr_v2t)

</details>



## Get stated

### Requirements 

* Python 3.7

- Pytorch 1.4.0 + 
- Transformers 3.1.0
- Numpy 1.18.1

```
pip install -r requirements.txt
```



### Training 

Training + evaluation:

```
python -m demo --config configs/$model_name/$dataset_$split_trainval.json
```

Evaluation from checkpoint:

```
python -m demo --config configs/$model_name/$dataset_$split_trainval.json --only_eval --load_checkpoint $checkpoint_path
```

Training from pretrained model:

```
python -m demo --config configs/$model_name/prtrn_$dataset_$split_trainval.json --load_checkpoint $checkpoint_path
```

Retrieval videos with a specific sentence:
```
python -m demo --config configs/$model_name/$dataset_$split_trainval.json --only_eval --load_checkpoint $checkpoint_path --sentence
```

Using the modified version of MMT for training:
```
python -m demo --config configs/$model_name/prtrn_$dataset_$split_trainval.json --modified_model
```
