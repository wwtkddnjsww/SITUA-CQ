# SITUA CQ

In this repository, we offer the code of the SITUA-CQ algorithm, 
which is published in IoT Journal [1] and an extended version of the research paper published in WCNC 2022 [2].

Moreover, this code is based on PyTorch and QPyTorch [3].

To use this code, please refer to SITUA-CQ under below.

'''
[1] S. Seo, J. Lee, H. Ko, and S. Pack, "Situation-Aware Cluster and Quantization Level Selection Algorithm for Fast Federated Learning," IEEE Internet of Things Journal, doi: 10.1109/JIOT.2023.3262582.
'''

## Description of components in a nutshell
In 'preliminary Experiments' folder, the effects of the over all distributions, 
distributions of quantization levels and the number of selected clusters 
can be simulated. At this simulation, each cluster in our code have one client. 

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
    
If you want to generate the non-iid data distribution for each client, 
please refer to 'src/distribution_generator.py'

## References

[2] S. Seo, J. Lee, H. Ko, and S. Pack, 
``Performance-Aware Client and Quantization Level Selection Algorithm 
for Fast Federated Learning,'' 
in *Proc. IEEE WCNC 2022*, April 2022.

[3] T. Zhang, Z. Lin, G. Yang, and C. De Sa, 
"QPyTorch: A Low-Precision Arithmetic Simulation Framework," 
arXiv preprint arXiv:1910.04540, October 2019.

