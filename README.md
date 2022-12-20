# MemInferenceAttack

This repository is built to reproduce some membership inference attacks (MIAs)<br>
It's written in Tensorflow 2.2.0<br>

## Dataset
Image dataset can be found in Tensorflow Datasets<br>
Non image dataset can be found in https://github.com/privacytrustlab/datasets.


## How to use the repository
1. train the models

```python
python ./trainModel.py --ndata 10000 --dataset cifar10 --model densenet --epoch 50 --batch_size 128 --lr 1e-3 
```
Then the trained model will be stored in ../models/target/{args.dataset}_{args.ndata}_{args.model}.tf<br>
one can use python files in /attacks to do the attack. 
```python
cd ./attacks
python lossBased.py --ndata 10000 --dataset cifar10 --model densenet
```
