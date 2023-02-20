## YOLO_project
Replication of the master thesis "YOLO-CFAR: a Novel CFAR Target Detection Method Based on YOLO" by Lin, Yu-Ting.

## YOLO-CFAR paper
This implementation is trying to replicate some of the results of the following master thesis:   \
林郁庭(2021)。基於YOLO之雷達目標偵測演算法。國立清華大學通訊工程研究所碩士論文，新竹市。取自 https://hdl.handle.net/11296/7n49t5 

### Abstract
Constant False Alarm Rate (CFAR) detection is a common target detection algorithm in radar systems. However, non-homogeneous scanerios, such as 
multi-target and clutter scanerios, can dramatically affect the CFAR target detection performance because of the erroneous noise level estimation. 

In order to imporve the CFAR detection performance in non-homogeneous scanerios, we propose a novel CFAR detection method, based on a deep learning 
model: You Only Look Once (YOLO), called YOLO-CFAR. The proposed CFAR scheme does not require to estimate the noise level and use deep learning model 
for object detection to detect targets in RD map. The possibility of error propagation caused by inaccurate noise level estimation decreased, thus 
getting better CFAR target detection performance.

In this paper, we not only introduce YOLO in CFAR target detection, but also use Dynamic Range Compression (DRC) to pre-precoess the input data and add
a Deep Neural Network (DNN) classifier to further improve the performance of YOLO-CFAR. Simulation results demonstrate that YOLO-CFAR outperforms other 
conventional CFAR schemes especially in non-homogeneous scanerios, furthermore, YOLO-CFAR can achieve real-time detection with 71 FPS.

## YOLOv3 references
I just copy and paste the code from the Internet and watch youtube tutorials to built the project.
- Redmon, Joseph and Farhadi, Ali, [YOLOv3: An Incremental Improvement](https://doi.org/10.48550/arXiv.1804.02767), April 8, 2018. 
- Ayoosh Kathuria, [Whats new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b), April, 23, 2018. 
- Aladdin Persson, [YOLOv3 from Scratch](https://www.youtube.com/watch?v=Grir6TZbc1M), Feb 23, 2021. 
- Sanna Persson, [YOLOv3 from Scratch](https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0), Mar 21, 2021. 

## Issues
- Dependency-related error
  - The virtual envs are summarized below:
  - My PC ```(Intel i7-8700 + Nvidia Geforce RTX 2060)```: 
    - env ```pt3.7``` with CUDA 
        ```python
        python==3.7.13
        numpy==1.19.2
        pytorch==1.7.1
        torchaudio==0.7.2
        torchvision==0.8.2
        pandas==1.2.1
        pillow==8.1.0 
        tqdm==4.56.0
        matplotlib==3.3.4
        albumentations==0.5.2
        ```
  - Lab PC ```(Intel i7-12700 + Nvidia Geforce RTX 3060 Ti)```: 
    - env ```pt3.7``` without CUDA
        ```python
        python==3.7.13
        numpy==1.21.6
        torch==1.13.1
        torchvision==0.14.1
        pandas==1.3.5
        pillow==9.4.0
        tqdm==4.64.1
        matplotlib==3.5.3
        albumentations==1.3.0
        ```
    - env ```pt3.8``` with CUDA
        ```python
        python==3.8.16
        numpy==1.23.5
        pytorch==1.13.1
        pytorch-cuda==11.7
        torchaudio==0.13.1             
        torchvision==0.14.1
        pandas==1.5.2
        pillow==9.3.0
        tqdm==4.64.1
        matplotlib==3.6.2
        albumentations==1.3.0
        ```
  - An annoying bug in ```dataset.py``` due to pytorch version
    - The code segment that contains potential bug (on line ```149``` and ```155```)
    ![](https://i.imgur.com/w5hUN05.png)
    ![](https://i.imgur.com/R7TKmAo.png)
    - ```scale_idx = anchor_idx // self.num_anchors_per_scale``` works fine on my PC, but on lab PC will get the following warning, so I naturally followed the suggestions and changed the syntax to ([```torch.div()```](https://pytorch.org/docs/stable/generated/torch.div.html))
        ```clike!
        UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch.
        ```
    - After following the suggestion and chage  the deprecated usage ```//``` we have: ```scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')```. This piece of code works fine on lab PC, under both env ```pt3.7``` and ```pt3.8```, but failed on my PC.
    - The error only occur on my PC, under env ```pt3.7```, but this env is the initial and stable one.
        ```clike
        Original Traceback (most recent call last):
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\worker.py", line 198, in _worker_loop
            data = fetcher.fetch(index)
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in fetch
            data = [self.dataset[idx] for idx in possibly_batched_index]
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in <listcomp>
            data = [self.dataset[idx] for idx in possibly_batched_index]
          File "d:\Datasets\YOLOv3-PyTorch\dataset.py", line 153, in __getitem__
            scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')
        TypeError: div() got an unexpected keyword argument 'rounding_mode'
        ```
  - Way to solve it:
    - First, try using ```torch.div()```. If it doesn't work, then change it back to ```//```.


