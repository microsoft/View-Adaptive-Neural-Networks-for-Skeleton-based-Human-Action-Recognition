
# View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition

[Code for our paper View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition](https://arxiv.org/pdf/1804.07453.pdf)

![image](https://github.com/lcl-2019/VA/tree/master/image/Flowchat.png)

### Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0
- [Keras](https://keras.io/) 2.1.2 
- [Theano](http://deeplearning.net/software/theano/) 1.0


### Data Preparation

We need to first dowload the [NTU](https://github.com/shahroudy/NTURGB-D) dataset

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
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
  publisher={IEEE}
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

