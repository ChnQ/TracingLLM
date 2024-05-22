<!-- <p align="center">
  <img src="assets/logo.png"  height=120>
</p> -->


### <div align="center">Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language Models<div> 

<div align="center">
<a href="https://arxiv.org/abs/2402.19465"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:TracingLLM&color=red&logo=arxiv"></a> &ensp;
</div>



## 🌈 Introduction
We are excited to present "Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language Models," a pioneering study on exploring trustworthiness in LLMs during pre-training. 
We explores five key dimensions of trustworthiness: reliability, privacy, toxicity, fairness, and robustness. 
By employing linear probing and extracting steering vectors from LLMs' pre-training checkpoints, the study aims to uncover the potential of pre-training in enhancing LLMs' trustworthiness. Furthermore, we investigates the dynamics of trustworthiness during pre-training through mutual information estimation, observing a two-phase phenomenon: fitting and compression. 
Our findings unveil new insights and encourage further developments in improving the trustworthiness of LLMs from an early stage.


![Overview Diagram](assets/overview.png)



##  🚩Features

We want to **ANSWER**: 

- How LLMs dynamically encode trustworthiness during pre-trainin?
- How to harness the pre-training period for more trustworthy LLMs?

We **FIND** that:

- After the early pre-training period, middle layer representations of LLMs have already developed *linearly separable patterns* about trustworthiness.
- Steering vectors extracted from pre-training checkpoints could *promisingly enhance the SFT model’s trustworthiness*.
- During the pretraining period of LLMs, there exist two distinct phases regarding trustworthiness: *fitting and compression*.

## 🚀Getting Started

### 🔧Installation
```
conda env create -f environment.yml
```
### 🌟Usage
> Tips: Before running the script, please replace the model storage path in `src/generate_activations.py`, `src/eval_trustworthiness.py` file with your actual model storage path

#### 1. Run the Probing Experiments (Section 2: Probing LLM Pre-training Dynamics in Trustworthiness)
```
cd src/
sh scripts/probing.sh
```
#### 2. Run the Steering Vector Experiments (Section 3: Controlling Trustworthiness via the Steering Vectors from Pre-training Checkpoints)
```
cd src/
sh scripts/steering.sh
```


## 📝License
Distributed under the Apache-2.0 License. See LICENSE for more information.

## 📖BibTeX
```
@article{qian2024towards,
  title={Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language Models},
  author={Qian, Chen and Zhang, Jie and Yao, Wei and Liu, Dongrui and Yin, Zhenfei and Qiao, Yu and Liu, Yong and Shao, Jing},
  journal={arXiv preprint arXiv:2402.19465},
  year={2024}
}
