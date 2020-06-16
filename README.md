# Few-Shot Class-Incremental Learning

ArXiv version: https://arxiv.org/abs/2004.10956

---

***Few-Shot Class-Incremental Learning*** **(FSCIL)** is a novel problem setting for incremental learning, where a unified classifier is incrementally learned for new classes with very few training samples. In this repository, we provide baseline benchmarks and codes for implementation.

## TOPology-preserving knowledge InCrementer (TOPIC)

The TOPIC framework for FSCIL is built with *neural gas* [1], a seminal algorithm that learns the topology of the data manifold in feature space via *competitive Hebbian learning* (CHL). Neural gas is capable of preserving the topology of any heterogenous, non-uniform manifold, making it perfect for FSCIL with imbalanced old/new classes.  The following animation shows the neural gas + CHL process. You may refer to an online demo of [2] [GNG](https://www.demogng.de/js/demogng.html?_3DV) for better understanding the topology learning of neural gas.

![image](https://github.com/xyutao/fscil/blob/master/results/ng.gif)

The TOPIC code will be released later after the commercial freezing period. 

[1] Martinetz Thomas and Schulten Klaus. A "neural-gas" network learns topologies. *Artificial Neural Networks*, 1991. <br />
[2] Bernd Fritzke. A growing neural gas network learns topologies. *Advances in neural information processing systems*, 1995. 

## FSCIL Benchmark Settings

We modify CIFAR100, miniImageNet and CUB200 datasets for FSCIL. For CIFAR100 and miniImageNet, we choose 60 out of 100 classes as the base classes and split the rest 40 classes into 8 incremental learning sessions, each of which has 5 classes and 5 training samples per class. While for CUB200, we choose 100 base classes. The lists of image indexes used in each training session are stored in the directory
> data/index_list

## Comparison Results

In the following tables, we provide detailed test accuracies of each method under different settings of benchmark datasets and CNN models. The reported results are the mean accuracies averaged over 10 runs.

"Ft-CNN" indicates the baseline finetuning method with only softmax cross-entropy loss. "Joint-CNN" indicates the upper bound that trains the model on the joint set of all encountered training sets. "iCaRL*","EEIL*" and "NCM*" are representative *class-incremental learning* methods, where we adapt them to FSCIL setting for comparison. 

These results are also reported in the figures of the paper.

![image](https://github.com/xyutao/fscil/blob/master/results/fig4.png)

### CIFAR100, QuickNet, *5-way 5-shot* (Fig.4 (a))

Method/Sessions | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9
-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
*Ft-CNN* | 57.78 | 13.09 | 6.40 | 3.00 | 2.24 | 2.86 | 1.46 | 2.47 | 2.00
*Joint-CNN* | *57.78* | *53.30* | *49.50* | *46.20* | *43.80* | *41.20* | *39.10* | *37.80* | *35.90*
iCaRL* | 57.78 | 46.31 | 33.79 | 28.59 | 24.98 | 21.33 | 19.07 | 17.05 | 16.25
EEIL* | 57.78 | 41.32 | 35.19 | 29.95 | 25.65 | 23.20 | 22.19 | 20.61 | 18.53
NCM* | 57.78 | 48.91 | 41.91 | 38.05 | 30.61 | 26.68 | 24.79 | 22.15 | 19.50
**Ours-AL** | **57.78** | **49.52** | **44.32** | **39.59** | **33.72** | **30.65** | **27.36** | **25.06** | **23.12**
**Ours-AL-MML** | **57.78** | **49.49** | **44.12** | **39.82** | **35.07** | **31.42** | **27.82** | **25.47** | **24.17**

### CIFAR100, ResNet18, *5-way 5-shot* (Fig.4 (b))
Method/Sessions | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9
-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
*Ft-CNN* | 61.31 | 27.22 | 16.37 | 6.08 | 2.54 | 1.56 | 1.93 | 2.60 | 1.40
*Joint-CNN* | *61.31* | *56.60* | *52.60* | *49.00* | *46.00* | *43.30* | *40.90* | *38.70* | *36.80*
iCaRL* | 61.31 | 46.32 | 42.94 | 37.63 | 30.49 | 24.00 | 20.89 | 18.80 | 17.21
EEIL* | 61.31 | 46.58 | 44.00 | 37.29 | 33.14 | 27.12 | 24.10 | 21.57 | 19.58
NCM* | 61.31 | 47.80 | 39.31 | 31.91 | 25.68 | 21.35 | 18.67 | 17.24 | 14.17
**Ours-AL** | **61.31** | **48.58** | **43.77** | **37.19** | **32.28** | **29.67** | **26.44** | **25.18** | **21.80**
**Ours-AL-MML** | **61.31** | **50.09** | **45.17** | **41.16** | **37.48** | **35.52** | **32.19** | **29.46** | **24.42**

### miniImageNet, QuickNet, *5-way 5-shot* (Fig.4 (c))
Method/Sessions | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9
-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
*Ft-CNN* | 50.71 | 11.38 | 2.27 | 2.56 | 1.57 | 2.12 | 2.24 | 2.67 | 1.89
*Joint-CNN* | *50.71* | *46.80* | *43.50* | *40.60* | *38.00* | *35.80* | *33.80* | *32.00* | *30.40*
iCaRL* | 50.71 | 37.55 | 31.65 | 26.49 | 23.33 | 20.75 | 17.08 | 14.69 | 11.05
EEIL* | 50.71 | 39.20 | 33.55 | 29.84 | 26.47 | 22.41 | 18.79 | 16.74 | 13.59
NCM* | 50.71 | 36.49 | 30.44 | 25.40 | 22.08 | 19.68 | 15.95 | 13.09 | 10.84
**Ours-AL** | **50.71** | **37.49** | **32.32** | **28.02** | **24.90** | **22.63** | **19.75** | **17.75** | **14.50**
**Ours-AL-MML** | **50.71** | **38.55** | **34.35** | **30.66** | **27.81** | **24.94** | **22.22** | **19.97** | **18.36**

### miniImageNet, ResNet18, *5-way 5-shot* (Fig.4 (d))
Method/Sessions | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9
-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
*Ft-CNN* | 61.31 | 27.22 | 16.37 | 6.08 | 2.54 | 1.56 | 1.93 | 2.60 | 1.40
*Joint-CNN* | *61.31* | *56.60* | *52.60* | *49.00* | *46.00* | *43.30* | *40.90* | *38.70* | *36.80*
iCaRL* | 61.31 | 46.32 | 42.94 | 37.63 | 30.49 | 24.00 | 20.89 | 18.80 | 17.21
EEIL* | 61.31 | 46.58 | 44.00 | 37.29 | 33.14 | 27.12 | 24.10 | 21.57 | 19.58
NCM* | 61.31 | 47.80 | 39.31 | 31.91 | 25.68 | 21.35 | 18.67 | 17.24 | 14.17
**Ours-AL** | **61.31** | **48.58** | **43.77** | **37.19** | **32.38** | **29.67** | **26.44** | **25.18** | **21.80**
**Ours-AL-MML** | **61.31** | **50.09** | **45.17** | **41.16** | **37.48** | **35.52** | **32.19** | **29.46** | **24.42**

### CUB200, ResNet18, *10-way 5-shot*

For CUB200, the comparison methods with their original learning rate settings have much worse test accuracy. We carefully tune their learning rates and greatly boost their original accuracy. In the table below, we use * to denote the settings with the improved accuracy.

Method/Sessions | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9 | 10 | 11
-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
*Ft-CNN* | 68.68 | 43.70 | 25.05 | 17.72 | 18.08 | 16.95 | 15.10 | 10.60 | 8.93 | 8.93 | 8.47
*Ft-CNN** | 68.68 | 44.81 | 32.26 | 25.83 | 25.62 | 25.22 | 20.84 | 16.77 | 18.82 | 18.25 | 17.18
*Joint-CNN* | *68.68* | *62.43* | *57.23* | *52.80* | *49.50* | *46.10* | *42.80* | *40.10* | *38.70* | *37.10* | *35.60*
iCaRL | 68.68 | 60.50 | 46.19 | 31.87 | 29.07 | 21.86 | 21.22 | 19.15 | 16.50 | 14.46 | 14.14
iCaRL* | 68.68 | 52.65 | 48.61 | 44.16 | 36.62 | 29.52 | 27.83 | 26.26 | 24.01 | 23.89 | 21.16
EEIL | 68.68 | 57.64 | 42.91 | 28.16 | 27.05 | 25.52 | 25.08 | 22.06 | 19.93 | 19.74 | 19.61
EEIL* | 68.68 | 53.63 | 47.91 | 44.20 | 36.30 | 27.46 | 25.93 | 24.70 | 23.95 | 24.13 | 22.11
NCM | 68.68 | 62.55 | 50.33 | 45.07 | 38.25 | 32.58 | 28.71 | 26.28 | 23.80 | 19.91 | 17.82
NCM* | 68.68 | 57.12 | 44.21 | 28.78 | 26.71 | 25.66 | 24.62 | 21.52 | 20.12 | 20.06 | 19.87
**Ours-AL** | **68.68** | **61.01** | **55.35** | **50.01** | **42.42** | **39.07** | **35.47** | **32.87** | **30.04** | **25.91** | **24.85**
**Ours-AL-MML** | **68.68** | **62.49** | **54.81** | **49.99** | **45.25** | **41.40** | **38.35** | **35.36** | **32.22** | **28.31** | **26.28**

## Following and Citing FSCIL

FSCIL is an unsolved, challenging but practical incremental learning setting. It still has large research potentials for new solutions and better performances. When you wish to conduct your research using this setting or refer to the baseline results in your paper, please use the following BibTeX entry.

```BibTeX
@inproceedings{tao2020fscil,
  author = {Tao, Xiaoyu and Hong, Xiaopeng and Chang, Xinyuan and Dong, Songlin and Wei, Xing and Gong, Yihong},
  title = {Few-Shot Class-Incremental Learning},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```
