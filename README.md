## YOLO_project
Replication of the master thesis "YOLO-CFAR: a Novel CFAR Target Detection Method Based on YOLO" 
林郁庭(2021)。基於YOLO之雷達目標偵測演算法。國立清華大學通訊工程研究所碩士論文，新竹市。 取自 https://hdl.handle.net/11296/7n49t5 

## YOLO-CFAR paper
The implementation is kinda based on the master thesis "YOLO-CFAR: a Novel CFAR Target Detection Method Based on YOLO" by Lin, Yu-Ting

### Abstract
Constant False Alarm Rate (CFAR) detection is a common target detection algorithm in radar systems. However, non-homogeneous scanerios, 
such as multi-target scanerios and clutter scanerios, can dramatically affect the CFAR target detection performance because of the erroneous 
noise level estimation. In order to imporve the CFAR detection performance in non-homogeneous scanerios, we propose a novel CFAR detection method, 
based on a deep learning model: You Only Look Once (YOLO), called YOLO-CFAR. The proposed CFAR scheme does not require to estimate the noise 
level and use deep learning model for object detection to detect targets in RD map. The possibility of error propagation caused by inaccurate 
noise level estimation decreased, thus getting better CFAR target detection performance.

In this paper, we not only introduce YOLO in CFAR target detection, but also use Dynamic Range Compression (DRC) to pre-precoess the input data and add
Deep Neural Network (DNN) to further improve the performance of YOLO-CFAR. Simulation results demonstrate that YOLO-CFAR outperforms other conventional
CFAR schemes especially in non-homogeneous scanerios, furthermore, YOLO-CFAR can achieve real-time detection with 71 FPS.
