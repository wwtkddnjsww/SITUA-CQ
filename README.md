# SITUA CQ

In this repository, we offer the code of SITUA-CQ algorithm, 
which is extended version of research paper published in WCNC 2022 [1].

Moreover, this code is based on PyTorch and QPyTorch [2].

## Description of components in a nutshell
In 'preliminary Experiments' folder, the effects of the over all distributions, 
distributions of quantization levels and the number of selected clusters 
can be simulated.

Files below can simulate:
1) 'src/preliminary Experiments/test_distribution_of_theta_b.py'
    - Effect of distributions of quantization levels

2) 'src/preliminary Experiments/test_non_IID.py'
    - Effect of the overall data distribution
    
3) 'src/preliminary Experiments/test_the_number_of_clients.py'
    - Effect of the number of selected clusters

Moreover, in 'Simulation Result Experiment' folder, 
SITUA-CQ algorithm can be simulated.

1) 'src/Simulation Result Experiment'
    - the effect of accuracy 

## References
[1] S. Seo, J. Lee, H. Ko, and S. Pack, 
``Performance-Aware Client and Quantization Level Selection Algorithm 
for Fast Federated Learning,'' 
in *Proc. IEEE WCNC 2022*, April 2022.

[2] T. Zhang, Z. Lin, G. Yang, and C. De Sa, 
"QPyTorch: A Low-Precision Arithmetic Simulation Framework," 
arXiv preprint arXiv:1910.04540, October 2019.

