# MemInferenceAttack

This repository is built to reproduce some MIA（membership inference attack）<br>
And it's written in pytorch<br>
All the papers we referred to are shown as below：<br>

## Attacks: <br>
•	[Membership inference attacks against machine learning models ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7958568)(Shokri et al., 2017 S&P)<br>
•	[Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models ](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-1_Salem_paper.pdf) (Salem et al., 2019 NDSS)<br>
•	[Practical Blind Membership Inference Attack via Differential Comparisons  
Defenses:](https://arxiv.org/abs/2101.01341) (Hui et al., 2021 NDSS)<br> 
•	[Machine Learning with Membership Privacy using Adversarial Regularization ](https://dl.acm.org/doi/pdf/10.1145/3243734.3243855)  (Nasr et al. 2019 CCS)<br>
•	[MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples](https://arxiv.org/abs/1909.10594) (Jia et al., 2019 CCS)

## Dataset
All the attacks are written in /attacks folder, we tried to wirte [defense](https://arxiv.org/abs/1909.10594) but failed to test on all dataset except on location dataset. Thus in /denfese folder, are codes we will work on but failed for now.


Our test dataset can be found in Tensorflow Datasets（torchvision）<br>
Non image dataset can be found in https://github.com/privacytrustlab/datasets.


## How to use the repository
One should train first, using trainModel.py (Following dataset can be viewed as an exmaple on cifar10 dataset)

```python
python ./trainModel.py --ndata 10000 --dataset cifar10 --model cnn --epoch 50 --batch_size 128 --lr 1e-3 
```
Then the trained model will be stored in ../models/target/{args.dataset}_{args.ndata}_{args.model}.tf<br>
one can use python files in /attacks to do the attack. 
```python
cd ./attacks
python BlindMidiffw.py --ndata 10000 --dataset cifar10 --model ResNet50 --gen 50 --test False
```
