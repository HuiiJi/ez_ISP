# ez_ISP : a easy_ISP for RAW to RGB conversion

![pipe](assets/ez_ISP.png)

if you want to read the Chinese version, please click [here](README_ch.md).
## Introduction

This is a easy ISP (aka:ez_ISP) for RAW to RGB conversion. It is based on the package of `numpy`, and it is easy to use and understand. The ez_ISP project is implemented by python, and it is easy to transplant to other platforms such as C/C++ for speed up. 

- [x] Bad Pixel Correction， 坏点校正
- [x] Black Level Correction， 黑电平校正
- [x] Anti Aliasing Filter， 抗混叠
- [x] Bayer Noise Reduction， RAW域去噪
- [x] Auto White Balance， 自动白平衡
- [x] Color Filter Array Interpolation， 去马赛克
- [x] Color Correction Matrix， 颜色矫正
- [x] Global Tone Mapping， 全局色调映射
- [x] Gamma Correction， Gamma映射
- [x] Edge Enhancement， 边缘增强
- [x] Brightness Contrast Control，亮度控制
- [x] Chorma Noise Reduction，Chorma域去噪
- [ ] Lens Shading Correction， 阴影矫正
- [ ] Luma Noise Reduction， Luma域去噪
- [ ] Local Tone Mapping， 局部色调映射

## File Structure

The ez_ISP project tree structure is listed as follows.
```shell
ez_ISP
│  .gitignore
│  run.py
|  isp_pipeline.py
│  LICENSE
│  README.md
│
├─config
│      isp_config.yaml
│
├─assets
│      raw.png
│
├─algorithm
|     __init__.py
|     aaf.py
|     awb.py
│     bcc.py
│     blc.py
│     bnr.py
│     bpc.py
│     ccm.py
│     cfa.py
│     cnr.py
│     ee.py
|     fir.py
|     gmc.py
|     gtm.py
|     ltm.py
|     r2y.py
│     utils.py
│     y2r.py
│
├─test_images
│      test.RAW
│
```
Device: AMD Ryzen 5 5600 6-Core Processor@4.20 GHz, Image Resolution: 1920x1080, Running time cost here:

|Module             |ez_ISP |
|:-----------------:|:------:|
|BPC                |2975.53 ms|
|BLC                |20.52 ms|
|AAF                |3932.32 ms|
|AWB                |19.02 ms |
|BNR                |73.99 ms|
|CFA                |11609.39 ms|
|CCM                |132.62 ms |
|GTM                |17.02 ms |
|GAC                |17.02 ms |
|R2Y                |111.10 ms |
|CNR                |3905.38 ms|
|EE                 |4068.87.1s |
|HSC                |56.34 ms|
|BBC                |9.01 ms |
|Total pipeline     |27.12 s |

Time cost: 27.12 s for a 1920x1080 image, though it is not fast enough, it is easy to use and easy to understand.

## Install
You can install ez_ISP by pip install the packages below.
- The main package is `numpy`, and `opencv-python` is used for image I/O. 
- Other packages are used for the demo such as `rawpy` and `yaml`, `time`, `os`.
```bash
pip install yaml, numpy, opencv-python, time, rawpy, os
```
Clone the ez_ISP project from github, and you can run the project.
```bash
git clone https://github.com/HuiiJi/ez_ISP.git
cd ez_ISP
```
Make sure that you have installed the packages above, or you will get an error when you run the project.

## How to use
The ez_ISP project is run by the `run.py` file.
```python
python run.py
```
![show](assets/running.jpg)
But before you run the py, please config the `config/isp_config.yaml` file, The config file is listed as follows.
```yaml
# -------------------- ISP Module Enable/Disable --------------------
enable:                
  BPC: True
  LCS: False                 # not implemented yet
  BLC: True
  AAF: True
  AWB: True
  BNR: False                 # not implemented yet
  CFA: True
  CCM: True
  GTM: True
  GMC: True
  R2Y: True
  CNR: True
  EE:  True
  BCC: True
  HSC: False                 # not implemented yet
  Y2R: True

# -------------------- Algorithm Params --------------------
RAW_img_path: '/mnt/cvisp/isp/ez_ISP/test_images/2DNR_Case_1_1.raw'
RAW_Height: 1080
RAW_Width: 1920
white_level: 1023
bayer_pattern: RGGB

BPC:
  bad_pixel_threshold: 30

LCS: ~

BLC:
  black_level_r: 256.0
  black_level_gr: 256.0
  black_level_gb: 256.0
  black_level_b: 256.0
  alpha: 1.        
  beta: 1.                  

AAF: ~

AWB:
  r_gain:  1.6             
  b_gain: 2.0            

BNR:
  BNR_method: 'bilateral'

CFA:
  CFA_method: 'bilinear'

CCM:
  ccm_matrix:
    - [1.631906, -0.381807, -0.250099]
    - [-0.298296, 1.614734, -0.316438]
    - [0.023770, -0.538501, 1.514732 ]

GTM:
  GTM_method: 'smoothstep'

GMC:
  gamma: 2.0

R2T: ~

CNR:
  CNR_method: 'gaussian'
  CNR_threshold: 0.3

EE:
  edge_enhancement_strength: 0.3

BCC:
  BCC_contrast: 0.1
  BCC_brightness: 10

HSC: ~

Y2R: ~
```
The params are listed as follows.
- `enable`: enable or disable the ISP module.
- `RAW_img_path`: the path of the RAW image.
- `RAW_Height`: the height of the RAW image.
- `RAW_Width`: the width of the RAW image.
- `white_level`: the white level of the RAW image.
- `bayer_pattern`: the bayer pattern of the RAW image.

If you don't want to use the ISP module, just set the `enable` to `False`. What you must to config is the `RAW_img_path`, `RAW_Height`, `RAW_Width`. The other params are the params of the ISP module, you can set them according to your needs.The result will be saved in `demo_outputs` folder.

## Course
Here are some courses about ISP, you can learn more about ISP from these courses.
### camera related
- [ISP Pipe1， 了解](https://web.stanford.edu/class/cs231m/lectures/lecture-11-camera-isp.pdf)
- [ISP Pipe2， 了解](http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/lectures/16_camerapipeline1.pdf)
- [ISP Pipe3， 了解](http://graphics.cs.cmu.edu/courses/15-463/2017_fall/lectures/lecture2.pdf)

You can learn more about ISP Pipe from these courses., but i think these courses are not useful for me, because i am not a camera engineer, i just want to learn the ISP algorithm, so i think the courses below are more useful for me.
### Computational photography
- [Camera History， 了解](http://www.cs.cornell.edu/courses/cs6640/2012fa/slides/01-History.pdf)
- [Recommend for ISP，推荐](https://www.eecs.yorku.ca/~mbrown/ICCV19_Tutorial_MSBrown.pdf)
- [Digital Photomontage，了解](https://graphics.stanford.edu/talks/compphot-publictalk-may08.pdf)
- [Marc大佬开课，强烈推荐](https://sites.google.com/site/marclevoylectures/schedule)

## Rerfence
Here are some open source ISP projects, which are very helpful for me to complete this project.
### Python

- [openISP](https://github.com/cruxopen/openISP)
- [fast-openISP](https://github.com/QiuJueqin/fast-openISP)
### C/C++

- [HDR-ISP](https://github.com/JokerEyeAdas/HDR-ISP)
- [ISPLab](https://github.com/yuqing-liu-dut/ISPLab)

## License
[MIT](https://choosealicense.com/licenses/mit/)
Thanks for your attention! 
If you have any questions, please contact me @HuiiJi.





