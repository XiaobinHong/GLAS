# GLAS: Gloving Label and Covariate Shift for Graph OOD Generalization
---

Graph Out-Of-Distribution (OOD) generalization aims to adapt a trained graph neural network (GNN) to unseen target graphs. Several methods have tried the invariance principle to improve the OOD generalization of GNNs. However, previous approaches typically assume that both source and target graphs have balanced label distribution. In the real world, it is natural that label distribution shifts as the domain changes. Unfortunately, the majority of existing graph OOD algorithms fail to address the coexistence of label and covariate shifts. We introduce a novel framework to tackle the challenges of graph OOD generalization by Governing Label and covAriate Shifts (GLAS). In training time, we present a class-calibrated invariant training method, incorporating a simple yet efficient latent augmentation strategy for training an effective GNN encoder and generating label-unbiased prototypes. During test time, we introduce a target distribution inspired TTA method, leveraging calibrated entropy minimization and a learnable consistency loss to align with the main task. In a joint effort during training and test times, GLAS significantly enhances the generalization performance of GNNs on unseen target graphs. Extensive experiments on 6 graph OOD benchmarks in comparison with 5 state-of-the-art graph OOD algorithms, validate the superiority and effectiveness of the proposed method.

Source code for GLAS (Gloving Label and Covariate Shift for Graph OOD Generalization)


## Installation
---
deeprobust==0.2.5 \
dgl==0.9.1\
dgl_cu102==0.6.1\
GCL==0.6.11\
googledrivedownloader==0.4\
ipdb==0.13.7\
matplotlib==3.5.2\
networkx==2.5\
numpy==1.20.1\
ogb==1.3.5\
pandas==1.2.3\
scikit_learn==1.1.3\
scipy==1.6.2\
torch==1.13.0\
torch_geometric==2.0.1\
torch_scatter==2.0.8\
torch_sparse==0.6.12\
tqdm==4.60.0\
visualization==1.0.0

## Data download
---
In our experiment, we consider three types of distribution shifts with six real-world datasets proposed by EERM (https://github.com/qitianwu/GraphOOD-EERM). 

You can make a directory ./data and download all the datasets through the Google drive:

``
https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing
``

And then fix the corresponding data directory under the GraphOOD-EERM

``
args.data_dir = YOUR_DATA_PATH
``

## Running the code
---
To run the code, you can use this script:

``
bash ./run1.sh
``

Or:

``
python main_train.py --dataset cora --gnn GCN --nlayers 2 --hidden 64 --run 10
``
