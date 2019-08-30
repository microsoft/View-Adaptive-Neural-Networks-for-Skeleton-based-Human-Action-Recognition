
# View adaptive neural networks for high performance skeleton-based human action recognition

Microsoft Open Source Code of Conduct: https://opensource.microsoft.com/codeofconduct


### Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0
- [Keras](https://keras.io/) 2.1.2 
- [Theano](http://deeplearning.net/software/theano/) 1.0


### Data Preparation

We need to first dowload the [NTU](https://github.com/shahroudy/NTURGB-D) dataset

- Extract the skeleton dataset to ./data/ntu
- Process the data
```bash
 cd ./data/ntu
 python get_raw_skes_data.py & python get_raw_denoised_data.py & python seq_transformation.py
```


### Training

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --VA --aug 1 --train 1

# For CNN-based model without view adaptation module
python  va-cnn.py --baseline --aug 1 --train 1

# For RNN-based model with view adaptation module
python va-rnn.py --VA --aug 1 --train 1

# For RNN-based model without view adaptation module
python va-rnn.py --baseline --aug 1 --train 1
```

### Reference
If you find our paper and repo useful, please cite our paper. Thanks!

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

