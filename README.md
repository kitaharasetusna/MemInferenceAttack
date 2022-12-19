# MemInferenceAttack

This repository is built to reproduce some MIA（membership inference attack）<br>
All the papers we referred to are shown as below：<br>

Attacks: <br>
•	[Membership inference attacks against machine learning models ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7958568)(Shokri et al., 2017 S&P)<br>
•	[Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models ](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-1_Salem_paper.pdf) (Salem et al., 2019 NDSS)<br>
•	[Practical Blind Membership Inference Attack via Differential Comparisons  
Defenses:](https://arxiv.org/abs/2101.01341) (Hui et al., 2021 NDSS)<br> 
•	[Machine Learning with Membership Privacy using Adversarial Regularization ](https://dl.acm.org/doi/pdf/10.1145/3243734.3243855)  (Nasr et al. 2019 CCS)<br>
•	[MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples](https://arxiv.org/abs/1909.10594) (Jia et al., 2019 CCS)


All the attacks are written in /attacks folder, we tried to wirte [defense](https://arxiv.org/abs/1909.10594) but failed to test on all dataset except on MNIST. Thus in /denfese folder, are codes we will work on but failed for now.
