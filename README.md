# Reinforcement Learning for Batch Bioprocess Optimization (non-smooth application) {Before in https://gitlab.com/Panos108/rl-with-nonsmooth}


Bioprocesses have received a lot of attention to produce clean and sustainable alternatives to fossil-based materials. However, they are generally difficult to optimize due to their unsteady-state operation modes and stochastic behaviours. Furthermore, biological systems are highly complex, therefore plant-model mismatch is often present. To address the aforementioned challenges we propose a Reinforcement learning based optimization strategy for batch processes. In this work, we applied the Policy Gradient method from batch-to-batch to update a control policy parametrized by a recurrent neural network. We assume that a preliminary process model is available, which is exploited to obtain a preliminary optimal control policy. Subsequently, this policy is updated based on measurements from the true plant. The approach was verified on a case study using a more complex process model for the true system embedded with adequate process disturbance. 

This application is work-example using the methods discribed in https://arxiv.org/abs/1904.07292
