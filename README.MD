# Multi-agent algorithms
## Environments
Based on Multiagent Particle Envs, which includes both cooperative and competitive scenarios. All of them have continuous state space and action space.
## Algorithms
* MADDPG
* QMIX
## Settings
The implementations in this projects focus on continuous action/state space. However, the Multiagent Particle Envs take discrete action space as default. Before testing these algorithms, you must set MPE to continuous version.
## Requirements
* Python 3.7.5
* PyTorch 1.3
* tensorboardX
* gym 0.10.5
* https://github.com/Steven-Ho/multiagent-particle-envs forked from https://github.com/openai/multiagent-particle-envs but designed for special usage in this project