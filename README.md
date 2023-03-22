# AOD-Net_pytorch
这是一篇使用pytorch框架的AOD-Net除雾网络，内部包含了：给图像添加雾、训练部分、torch推理以及onnxruntime推理部分、pth2onnx模块

# AODNet-Based-Image-Haze-Removal
Single Image Haze Removal Using AODNet in Pytorch
* Implementation of Boyi Li's Paper [**An All-in-One Network for Dehazing and Beyond**](https://arxiv.org/pdf/1707.06543.pdf) on ICCV 2017.<br>

---
#### Contents

1. [Dependency](#Dependency)
1. [Usage](#Usage)
1. [Results](#Results)
1. [References](#References)
---

#### Dependency
###### &emsp;Python&ensp;3.6 or newer<br>
###### &emsp;torch == 1.7.1<br>
###### &emsp;pillow == 5.1.0<br>
###### &emsp;numpy == 1.14.3<br>
###### &emsp;matplotlib == 2.2.2<br>

#### Usage
* How to Use : download the whole project and run **inference.py**
* folder ./saved_models : where the trained models are saved, files are in .pth format.
* folder ./data/gt : groundtruth (haze free images) of the [**training data**](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat).
* folder ./data/hazy : corresponding hazy images of the training data.
* folder ./test_images : some testing images that appear in the original paper.
* data.py : function that loads the training data.
* train.py : train a new AODNet from scratch using training data saved in folder ./data/.
* model.py : definition of AODNet.
* utils.py : some auxiliary functions.
* inference.py : single image dehazing using the trained AODNet.

#### Results
* [**Tiananmen Square**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result0.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result0.png /><br><br>

* [**Mountain View**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result1.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result1.png /><br><br>

* [**Toys**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result3.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result3.png /><br><br>

* [**Landscape**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result4.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result4.png /><br><br>

* [**Indoor**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result7.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result7.png /><br><br>

* [**Foggy Train**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result5.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result5.png /><br><br>

* [**The Forbidden City**](https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result6.png)<br>
<img src= https://github.com/ForeverPs/AODNet-Based-Image-Haze-Removal/blob/main/results/result6.png /><br><br>


#### References
##### Author&ensp;:&ensp;Boyi Li, Xiulian Peng, Zhangyang Wang, Jizheng Xu, Dan Feng<br>
* ###### &ensp;[An All-in-One Network for Dehazing and Beyond----ICCV 2017](https://arxiv.org/pdf/1707.06543.pdf)<br>

