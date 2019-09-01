
# View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition

This repository hold the codes and methods for the following papers:

**View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition**. TPAMI, 2019, [paper][https://arxiv.org/pdf/1804.07453.pdf]

**View Adaptive Recurrent Neural Networks for High Performance Human Action Recognition from Skeleton Data**. ICCV, 2017, [paper][http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.pdf]

### Flowchart

![image](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/blob/master/image/Flowchart.png)

Figure 1: Flowchat of the end-to-end view adaptive neural network. It consists of a main classification network and a view adaptation subnetwork. The view adaptation subnetwork automatically determines the virtual observation viewpoints and transforms the skeleton input to representations under the new viewpoints for classification by the main classification network. The entire network is end-toend trained to optimize the classification performance.

Introduction: Skeleton-based human action recognition has recently attracted increasing attention thanks to the accessibility and the popularity of 3D skeleton data. One of the key challenges in skeleton-based action recognition lies in the large view variations when capturing data. In order to alleviate the effects of view variations, this paper introduces a novel view adaptation scheme, which automatically determines the virtual observation viewpoints in a learning based data driven manner. We design two view adaptive neural networks, i.e., VA-RNN based on RNN, and VA-CNN based on CNN. For each network, a novel view adaptation module learns and determines the most suitable observation viewpoints, and transforms the skeletons to those viewpoints for the end-to-end recognition with a main classification network. Ablation studies find that the proposed view adaptive models are capable of transforming the skeletons of various viewpoints to much more consistent virtual viewpoints which largely eliminates the viewpoint influence. In addition, we design a two-stream scheme (referred to as VA-fusion) that fuses the scores of the two networks to provide the fused prediction. Extensive experimental evaluations on five challenging benchmarks demonstrate that the effectiveness of the proposed view-adaptive networks and superior performance over state-of-the-art approaches.

### Framework

![image](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/blob/master/image/Framework.png)

Figure 2: Architecture of the proposed view adaptive neural networks: a view adaptive RNN with LSTM (VA-RNN), and a view adaptive CNN (VA-CNN). The classification scores from the two networks can be fused to provide the fused prediction, denoted as the VA-fusion scheme. Note that based on application requirements, we can use VA-RNN or VA-CNN only or combine them together. 

### Visualization of the Learned Views

![image](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/blob/master/image/visulization.png)

Figure 3: Frames of (a) the similar posture captured from different viewpoints for the same subject, and (b) the same action “drinking” captured from different viewpoints for different subjects. 2nd row: original skeletons. 3rd row: Skeleton representations from the observation viewpoints of our VA-RNN model. 4th row: Skeleton representations from the observation viewpoints of our VA-CNN model.


### Prerequisites
The code is built with following libraries:
- Python 2.7
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/) 1.0
- [Keras](https://keras.io/) 2.1.2 
- [Theano](http://deeplearning.net/software/theano/) 1.0


### Data Preparation

We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset

- Extract the dataset to ./data/ntu/nturgb+d_skeletons/
- Process the data
```bash
 cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


### Training

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 1

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 1

# For RNN-based model with view adaptation module
python va-rnn.py --model VA --aug 1 --train 1

# For RNN-based model without view adaptation module
python va-rnn.py --model baseline --aug 1 --train 1
```



### Testing

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 0

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 0

# For RNN-based model with view adaptation module
python va-rnn.py --model VA --aug 1 --train 0

# For RNN-based model without view adaptation module
python va-rnn.py --model baseline --aug 1 --train 0
```

### Reference
If you find our paper and repo useful, please cite our papers. Thanks!

```
@article{zhang2019view,
  title={View adaptive neural networks for high performance skeleton-based human action recognition},
  author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019},
}

@inproceedings{zhang2017view,
  title={View adaptive recurrent neural networks for high performance human action recognition from skeleton data},
  author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2117--2126},
  year={2017}
}

```
Microsoft Open Source Code of Conduct: https://opensource.microsoft.com/codeofconduct

