#  DecloudNet: Cross-patch Consistency is a Non-trivial Problem for Thin Cloud Removal from Wide-swath Multi-spectral Images
 Official implementation.

---

by Mingkai Li, Qizhi Xu, Jian Guo and Wei Li. 
Beijing Institute of Technology

### Introduction

![Alt text](figs/1.png)

Cloud cover leads to great loss of spatial details in wide-swath multispectral images, and thus significantly affects their application value. Wide-swath images are huge in size and are usually cropped into patches before thin cloud removal. Additionally, wide-swath images contain a rich variety of types and shapes of thin cloud, with each patch containing different clouds. However, most of the existing methods and datasets were primarily designed for natural image dehazing. These datasets had limited types and shapes of clouds. When applied to remote sensing image thin cloud removal task, these methods are unable to remove various types of clouds and lead to severe cross-patch color difference. To address this problem, a DecloudNet based on cross-patch consistency supervision was proposed. First, a Multi-sacle Cloud Perception Block (MCPB) with multi-size convolutional kernels was proposed to enhance the network's capability to extract clouds feature of different sizes. Second, a cross-patch consistency supervision was designed to reduce the network's inconsistent cloud removal strength in different patches and remove cross-patch color difference when processing wide-swath images. Finally, a thin cloud simulation method based on Perlin Noise, Domain Warping and Atmospheric Scattering Model was proposed to construct a high-quality declouding dataset containing clouds of multiple sizes and shapes, which can improve the performance of DecloudNet on different kinds of thin cloud. The DecloudNet and compared methods were tested for simulated thin cloud removal performance on images from QuickBird, GaoFen-2 and WorldView-2 satellites, and for real thin cloud removal performance on wide-swath images from GaoFen-1 Wide Field of View, GaoFen-1 and EarthObserving-1 satellites. The experiment results demonstrated that DecloudNet outperformed the existing State-Of-The-Art (SOTA) methods. DecloudNet and cross-patch consistency supervision made it possible to perform thin cloud removal on wide-swath images of large size on most GPUs without worrying about graphics memory limitation.

![Alt text](figs/2.png)

### Usage
#### Test

Trained_models are available at google drive: https://drive.google.com/file/d/1H-iEAT94sYso07W4PDmEOWhGU9gZ-Xgy/view?usp=sharing

*Put  models in the `trained_models/` folder.*

*Test images are in `test_imgs/` folder.*

 ```shell
 python test.py 
```
The training code, thin cloud simulation code and dataset will be available after the paper is accepted.
