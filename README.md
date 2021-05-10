# Awesome Graph Self-Supervised Learning

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)![GitHub stars](https://img.shields.io/github/stars/LirongWu/awesome-graph-self-supervised-learning?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/LirongWu/awesome-graph-self-supervised-learning?color=green&label=Fork)  ![visitors](https://visitor-badge.glitch.me/badge?page_id=LirongWu.awesome-graph-self-supervised-learning)





## Training Strategy

Considering the relationship among bottleneck encoders, self-supervised pretext tasks, and downstream tasks, the training strategies can be divided into three categories: Pre-training and Fine-tuning (P\&F), Joint Learning (JL), and Unsupervised Representation Learning (URL), with their detailed workflow shown below.

<p align="center">
  <img src='./figs/training strategy.PNG' width="500">
</p>




- Pre-train\&Fine-tune (P&F): it first pre-trains the encoder with unlabeled nodes by the self-supervised pretext tasks. The pre-trained encoder’s parameters are then used as the initialization of the encoder used in supervised fine-tuning for downstream tasks.
- Joint Learning (JL): an auxiliary pretext task with self-supervision is included to help learn the supervised downstream task. The encoder is trained through both the pretext task and the downstream task simultaneously.
- Unsupervised Representation Learning (URL): it first pre-trains the encoder with unlabeled nodes by the self-supervised pretext tasks. The pre-trained encoder’s parameters are then frozen and used in the supervised downstream task with additional labels.





## Contrastive Learning

#### Global-Global (Same-Scale)  Contrasting: 5 Papers

- GraphCL: Graph Contrastive Learning with Augmentations.
  - Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen. *NIPS 2020*. [[pdf]](https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) [[code]](https://github.com/Shen-Lab/GraphCL)
- IGSD: Iterative Graph Self-Distillation.
  - H. Zhang, S. Lin, W. Liu, P. Zhou, J. Tang, X. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2010.12609.pdf)
- DACL: Towards Domain-Agnostic Contrastive Learning.
  - V. Verma, M.-T. Luong, K. Kawaguchi, H. Pham, andQ. V. Le. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2011.04419.pdf)
- LCL: Label Contrastive Coding Based Graph Neural Network for Graph Classification.
  - Y. Ren, J. Bai, and J. Zhang. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2101.05486.pdf) [[code]](https://github.com/YuxiangRen/Label-Contrastive-Coding-based-Graph-Neural-Network-for-Graph-Classification-)
- CSSL: Contrastive Self-Supervised Learning for Graph Classification.
  - J. Zeng and P. Xie. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.05923.pdf)

#### Context-Context (Same-Scale) Contrasting: 1 Papers

- GCC: Graph Contrastive Coding for Graph Neural Network Pre-training.
  - J. Qiu, Q. Chen, Y. Dong, J. Zhang, H. Yang, M. Ding, K. Wang, and J. Tang. *KDD 2020*. [[pdf]](https://arxiv.org/pdf/2006.09963.pdf) [[code]](https://github.com/THUDM/GCC)

#### Local-Local (Same-Scale) Contrasting: 12 Papers

- GRACE: Deep Graph Contrastive Representation Learning.
  - Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.04131.pdf) [[code]](https://github.com/CRIPAC-DIG/GRACE)
- GCA: Graph Contrastive Learning with Adaptive Augmentation.
  - Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2010.14945.pdf) [[code]](https://github.com/CRIPAC-DIG/GCA)
- GROC: Towards Robust Graph Contrastive Learning.
  - N. Jovanovi´c, Z. Meng, L. Faber, and R. Wattenhofer. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2102.13085.pdf)
- STDGI: Spatio-Temporal Deep Graph Infomax.
  - F. L. Opolka, A. Solomon, C. Cangea, P. Veliˇckovi´c, P. Li` o, and R. D. Hjelm. *Arxiv 2019*. [[pdf]](https://arxiv.org/pdf/1904.06316.pdf)
- GMI: Graph Representation Learning via Graphical Mutual Information Maximization.
  - L. Yu, S. Pei, C. Zhang, L. Ding, J. Zhou, L. Li, and X. Zhang. *WWW 2020*. [[pdf]](https://arxiv.org/pdf/2002.01169.pdf) [[code]](https://github.com/zpeng27/GMI)
- KS2L: Self-Supervised Smoothing Graph Neural Networks.
  - L. Yu, S. Pei, C. Zhang, L. Ding, J. Zhou, L. Li, and X. Zhang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.00934.pdf)
- CG3: Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning.
  - S. Wan, S. Pan, J. Yang, and C. Gong. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.07111.pdf)
- BGRL: Bootstrapped Representation Learning on Graphs.
  - S. Thakoor, C. Tallec, M. G. Azar, R. Munos, P. Veliˇckovi´c, and M. Valko. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2102.06514.pdf)
- SelfGNN: Self-supervised Graph Neural Networks without Explicit Negative Sampling.
  - Z. T. Kefato and S. Girdzijauskas. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2103.14958.pdf) [[code]](https://github.com/zekarias-tilahun/SelfGNN)
- PT-DGNN: Pre-training on Dynamic Graph Neural Networks.
  - J. Zhang, K. Chen, and Y. Wang. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2102.12380.pdf) [[code]](https://github.com/Mobzhang/PT-DGNN)
- COAD: Coad: Contrastive Pretraining with Adversarial Fine-tuning for Zero-shot Expert Linking.
  - B. Chen, J. Zhang, X. Zhang, X. Tang, L. Cai, H. Chen, C. Li, P. Zhang, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2012.11336.pdf) [[code]](https://github.com/allanchen95/Expert-Linking)
- Contrast-Reg: Improving Graph Representation Learning by Contrastive Regularization.
  - K. Ma, H. Yang, H. Yang, T. Jin, P. Chen, Y. Chen, B. F. Kamhoua, and J. Cheng. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2101.11525.pdf)

#### Local-Global (Cross-Scale) Contrasting: 5 Papers

- DGI: Deep Graph Infomax.
  - P. Velickovic, W. Fedus, W. L. Hamilton, P. Li` o, Y. Bengio, and R. D. Hjelm. *ICLR 2019*. [[pdf]](https://arxiv.org/pdf/1809.10341.pdf) [[code]](https://github.com/PetarV-/DGI)
- HDMI: Hdmi: High-order Deep Multiplex Infomax.
  - B. Jing, C. Park, and H. Tong. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2102.07810.pdf)
- DMGI: Unsupervised Attributed Multiplex Network Embedding.
  - C. Park, D. Kim, J. Han, and H. Yu. *AAAI 2020*. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/5985/5841) [[code]](https://github.com/pcy1302/DMGI)
- MVGRL: Contrastive Multi-View Representation Learning on Graphs.
  - K. Hassani and A. H. K. Ahmadi. *ICML 2020*. [[pdf]](http://proceedings.mlr.press/v119/hassani20a/hassani20a.pdf) [[code]](https://github.com/kavehhassani/mvgrl)
- HIGI: Heterogeneous Deep Graph Infomax.
  - Y. Ren, B. Liu, C. Huang, P. Dai, L. Bo, and J. Zhang. *Arxiv 2019*. [[pdf]](https://arxiv.org/pdf/1911.08538.pdf) [[code]](https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax)

#### Local-Context (Cross-Scale) Contrasting: 6 Papers

- Subg-Con: Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning.
  - Y. Jiao, Y. Xiong, J. Zhang, Y. Zhang, T. Zhang, and Y. Zhu. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.10273.pdf) [[code]](https://github.com/yzjiao/Subg-Con)
- Strategies for Pre-training Graph Neural Networks.
  - W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. S. Pande, and J. Leskovec. *ICLR 2020*. [[pdf]](https://arxiv.org/pdf/1905.12265.pdf) [[code]](http://snap.stanford.edu/gnn-pretrain)
- GIC: Leveraging Cluster-level Node Information for Unsupervised Graph Representation Learning.
  - C. Mavromatis and G. Karypis. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.06946.pdf) [[code]](https://github.com/cmavro/Graph-InfoClust-GIC)
- GraphLoG: Self-Supervised Graph-level Representation Learning with Local and Global Structure.
  - M. Xu, H. Wang, B. Ni, H. Guo, and J. Tang. *OpenReview 2021*. [[pdf]](https://openreview.net/forum?id=DAaaaqPv9-q) [[code]](https://openreview.net/forum?id=DAaaaqPv9-q)
- MHCN: Self-Supervised Multi-channel Hypergraph Convolutional Network for Social Recommendation.
  - J. Yu, H. Yin, J. Li, Q. Wang, N. Q. V. Hung, and X. Zhang. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2101.06448.pdf) [[code]](https://github.com/Coder-Yu/RecQ)
- EGI: Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization.
  - Q. Zhu, Y. Xu, H.Wang, C. Zhang, J. Han, and C. Yang. *Arxiv 2020*. [[pdf]](https://arxiv.org/abs/2009.05204) [[code]](https://openreview.net/forum?id=J_pvI6ap5Mn)

#### Context-Global (Cross-Scale) Contrasting: 5 Papers

- MICRO-Graph: Motif-Driven Contrastive Learning of Graph Representations.
  - S. Zhang, Z. Hu, A. Subramonian, and Y. Sun. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2012.12533.pdf) [[code]](https://drive.google.com/file/d/1b751rpnV-SDmUJvKZZI-AvpfEa9eHxo9/)
- InfoGraph: Unsupervised and Semi-Supervised Graph-level Representation Learning via Mutual Information Maximization.
  - F. Sun, J. Hoffmann, V. Verma, and J. Tang. *ICLR 2020*. [[pdf]](https://arxiv.org/pdf/1908.01000.pdf) [[code]](https://github.com/fanyun-sun/InfoGraph)
- SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism.
  - Q. Sun, H. Peng, J. Li, J. Wu, Y. Ning, P. S. Yu, and L. He. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2101.08170.pdf) [[code]](https://github.com/RingBDStack/SUGAR)
- BiGI: Bipartite Graph Embedding via Mutual Information Maximization.
  - J. Cao, X. Lin, S. Guo, L. Liu, T. Liu, and B. Wang. *WSDM 2021*. [[pdf]](https://arxiv.org/abs/1505.05192) [[code]](https://github.com/clhchtcjj/BiNE)
- HTC: Graph Representation Learning by Ensemble Aggregating Subgraphs via Mutual Information Maximization.
  - C. Wang and Z. Liu. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2103.13125.pdf)

## Generative Learning

####  Graph Autoencoding: 7 Papers

- Graph Completion: When Does Self-Supervision Help Graph Convolutional Networks?
  - Y. You, T. Chen, Z. Wang, and Y. Shen. *PMLR 2020*. [[pdf]](http://proceedings.mlr.press/v119/you20a/you20a.pdf) [[code]](https://github.com/Shen-Lab/SS-GCNs)
- Node Attribute Masking: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- Edge Attribute Masking: Strategies for Pre-training Graph Neural Networks.
  - W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. S. Pande, and J. Leskovec. *ICLR 2020*. [[pdf]](https://arxiv.org/pdf/1905.12265.pdf) [[code]](http://snap.stanford.edu/gnn-pretrain)
- Node Attribute and Embedding Denoising: Graph-based Neural Network Models with Multiple Self-Supervised Auxiliary Tasks.
  - F. Manessi and A. Rozza. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2011.07267.pdf)
- Adjacency Matrix Reconstruction: Self-Supervised Training of Graph Convolutional Networks.
  - Q. Zhu, B. Du, and P. Yan. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.02380.pdf)
- Graph Bert: Only Attention is Needed for Learning Graph Representations.
  - J. Zhang, H. Zhang, C. Xia, and L. Sun. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2001.05140.pdf) [[code]](https://github.com/anonymous-sourcecode/Graph-Bert)
- Pretrain-Recsys: Pretraining Graph Neural Networks for Cold-start Users and Items Representation.
  - B. Hao, J. Zhang, H. Yin, C. Li, and H. Chen. *WSDM 2021*. [[pdf]](https://dl.acm.org/doi/abs/10.1145/3437963.3441738) [[code]](https://github.com/jerryhao66/Pretrain-Recsys)

####  Graph Autoregression: 1 Papers

- GPT-GNN: Generative Pre-training of Graph Neural Networks.
  - Z. Hu, Y. Dong, K. Wang, K. Chang, and Y. Sun. *KDD 2020*. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403237) [[code]](https://github.com/acbull/GPT-GNN)

## Predictive Learning

####  Node Property Prediction: 1 Papers

- Node Property Prediction: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)

####  Context-based Prediction: 12 Papers

- S2GRL: Self-Supervised Graph Representation Learning via Global Context Prediction.
  - Z. Peng, Y. Dong, M. Luo, X.-M. Wu, and Q. Zheng. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2003.01604.pdf)
- PairwiseDistance: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- PairwiseAttsim: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- Distance2Cluster: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- EdgeMask: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- TopoTER: Unsupervised Learning of Topology Transformation Equivariant Representations.
  - X. Gao, W. Hu, and G.-J. Qi. *OpenReview 2021*. [[pdf]](https://openreview.net/forum?id=9az9VKjOx00)
- Centrality Score Ranking: Pretraining Graph Neural Networks for Generic Structural Feature Extraction.
  - Z. Hu, C. Fan, T. Chen, K.-W. Chang, and Y. Sun. *Arxiv 2019*.  [[pdf]](https://arxiv.org/pdf/1905.13728.pdf)
- Meta-path prediction: Self-supervised Auxiliary Learning with Meta-paths for Heterogeneous Graphs.
  - D. Hwang, J. Park, S. Kwon, K. Kim, J. Ha, and H. J. Kim. *NIPS 2020*. [[pdf]](https://arxiv.org/pdf/2007.08294.pdf) [[code]](https://github.com/mlvlab/SELAR)
- SLiCE: Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks.
  - P. Wang, K. Agarwal, C. Ham, S. Choudhury, and C. K. Reddy. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2007.11192.pdf) [[code]](https://github.com/pnnl/SLICE)
- Distance2Labeled: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- Distance2Labeled: Self-Supervised Learning on Graphs: Deep Insights and New Direction.
  - W. Jin, T. Derr, H. Liu, Y. Wang, S. Wang, Z. Liu, and J. Tang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2006.10141.pdf) [[code]](https://github.com/ChandlerBang/SelfTask-GNN)
- HTM: Hop-count based Self-Supervised Anomaly Detection on Attributed Networks.
  - T. Huang, Y. Pei, V. Menkovski, and M. Pechenizkiy. *Arxiv 2021*. [[pdf]](https://arxiv.org/pdf/2104.07917.pdf)

####  Self-Training: 5 Papers

- Multi-stage Self-training: Deeper insights into Graph Convolutional Networks for Semi-Supervised Learning.
  - Q. Li, Z. Han, and X. Wu. *AAAI 2018*. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/11604) [[code]](https://github.com/Davidham3/deeper_insights_into_GCNs)
- Node Clustering and Partitioning: When Does Self-Supervision Help Graph Convolutional Networks.
  - Y. You, T. Chen, Z. Wang, and Y. Shen. *PMLR 2020*. [[pdf]](http://proceedings.mlr.press/v119/you20a/you20a.pdf) [[code]](https://github.com/Shen-Lab/SS-GCNs)
- CAGAN: Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning.
  - Y. Zhu, Y. Xu, F. Yu, S. Wu, and L. Wang. *Arxiv 2020*. [[pdf]](https://arxiv.org/pdf/2009.01674.pdf)
- M3S: Multi-stage Self-Supervised Learning for Graph Convolutional Networks on Graphs with Few Labeled Nodes.
  - K. Sun, Z. Lin, and Z. Zhu. *AAAI 2020*. [[pdf]](https://deepai.org/publication/multi-stage-self-supervised-learning-for-graph-convolutional-networks) [[code]](https://github.com/datake/M3S)
- Cluster Preserving: Pretraining Graph Neural Networks for Generic Structural Feature Extraction.
  - Z. Hu, C. Fan, T. Chen, K.-W. Chang, and Y. Sun. *Arxiv 2019*. [[pdf]](https://arxiv.org/pdf/1905.13728.pdf)

####  Domain Knowledge-based Prediction: 2 Papers

- Contextual Molecular Property Prediction: Self-Supervised Graph Transformer on Large-Scale Molecular Data.
  - Y. Rong, Y. Bian, T. Xu, W. Xie, Y. Wei, W. Huang, and J. Huang. *NIPS 2020*. [[pdf]](https://drug.ai.tencent.com/publications/GROVER.pdf) [[code]](https://github.com/tencent-ailab/grover)
- Graph-level Motif Prediction: Self-Supervised Graph Transformer on Large-scale Molecular Data.
  - Y. Rong, Y. Bian, T. Xu, W. Xie, Y. Wei, W. Huang, and J. Huang. *NIPS 2020*. [[pdf]](https://drug.ai.tencent.com/publications/GROVER.pdf) [[code]](https://github.com/tencent-ailab/grover)

## A Summary of Methodology Details

About Graph Property, Pretext Task, Data Augmentation, Objective Function, Training Strategy, and Year of publication.

| Methods                                      |  Graph Property  |   Pretext-Task    |                      Data Augmentation                       |                  Objective Function                   | Training Strategy | Year |
| :------------------------------------------- | :--------------: | :---------------: | :----------------------------------------------------------: | :---------------------------------------------------: | :---------------: | :--: |
| Graph Completion                             |    Attributed    |   Generative/AE   |                      Attribute Masking                       |                          MAE                          |      P\&F/JL      | 2020 |
| Node Attribute Masking                       |    Attributed    |   Generative/AE   |                      Attribute Masking                       |                          MAE                          |      P\&F/JL      | 2020 |
| Edge Attribute Masking                       |    Attributed    |   Generative/AE   |                      Attribute Masking                       |                          MAE                          |       P\&F        | 2019 |
| Node Attribute and<br/>Embedding Denoising   |    Attributed    |   Generative/AE   |                      Attribute Masking                       |                          MAE                          |        JL         | 2020 |
| Adjacency Matrix Reconstruction              |    Attributed    |   Generative/AE   |           Attribute Masking<br/>Edge Perturbation            |                      MAE<br/>CE                       |        JL         | 2020 |
| Graph Bert                                   |    Attributed    |   Generative/AE   |           Attribute Masking<br/>Edge Perturbation            |                          MAE                          |       P\&F        | 2020 |
| Pretrain-Recsys                              |    Attributed    |   Generative/AE   |                      Edge Perturbation                       |                          MAE                          |       P\&F        | 2021 |
| GPT-GNN                                      |  Heterogeneous   |   Generative/AR   |           Attribute Masking<br/>Edge Perturbation            |                    MAE<br/>InfoNCE                    |       P\&F        | 2020 |
| GraphCL                                      |    Attributed    |  Contrastive/G-G  | Attribute Masking<br/>Edge Perturbation<br/>Random Walk Sampling |                        InfoNCE                        |        URL        | 2020 |
| IGSD                                         |    Attributed    |  Contrastive/G-G  |             Edge Perturbation<br/>Edge Doffisopm             |                        InfoNCE                        |      JL/URL       | 2020 |
| DACL                                         |    Attributed    |  Contrastive/G-G  |                            Mixup                             |                        InfoNCE                        |        URL        | 2020 |
| LCL                                          |    Attributed    |  Contrastive/G-G  |                             None                             |                        InfoNCE                        |        JL         | 2021 |
| CSSL                                         |    Attributed    |  Contrastive/G-G  |   NodeInsertion<br/>Edge Perturbation<br/>Uniform Sampling   |                        InfoNCE                        |    P\&F/JL/URL    | 2020 |
| GCC                                          |   Unattributed   |  Contrastive/C-C  |                     Random Walk Sampling                     |                        InfoNCE                        |     P\&F/URL      | 2020 |
| GRACE                                        |    Attributed    |  Contrastive/L-L  |           Attribute Masking<br/>Edge Perturbation            |                        InfoNCE                        |        URL        | 2020 |
| GCA                                          |    Attributed    |  Contrastive/L-L  |                       Attention-based                        |                        InfoNCE                        |        URL        | 2020 |
| GROC                                         |    Attributed    |  Contrastive/L-L  |                        Gradient-based                        |                        InfoNCE                        |        URL        | 2021 |
| STDGI                                        | Spatial-Temporal |  Contrastive/L-L  |                     Attribute Shuffling                      |                     JS Estimator                      |        URL        | 2019 |
| GMI                                          |    Attributed    |  Contrastive/L-L  |                             None                             |                     SP Estimator                      |        URL        | 2020 |
| KS2L                                         |    Attributed    |  Contrastive/L-L  |                             None                             |                        InfoNCE                        |        URL        | 2020 |
| CG3                                          |    Attributed    |  Contrastive/L-L  |                             None                             |                        InfoNCE                        |        JL         | 2020 |
| BGRL                                         |    Attributed    |  Contrastive/L-L  |           Attribute Masking<br/>Edge Perturbation            |                     Inner Product                     |        URL        | 2021 |
| SelfGNN                                      |    Attributed    |  Contrastive/L-L  |             Attribute Masking<br/>Edge Diffusion             |                          MSE                          |        URL        | 2021 |
| PT-DGNN                                      |     Dynamic      |  Contrastive/L-L  |           Attribute Masking<br/>Edge Perturbation            |                       InforNCE                        |       P\&F        | 2021 |
| COAD                                         |    Attributed    |  Contrastive/L-L  |                             None                             |                     Triplet Loss                      |       P\&F        | 2020 |
| Contrst-Reg                                  |    Attributed    |  Contrastive/L-L  |                     Attribute Shuffling                      |                        InfoNCE                        |        JL         | 2021 |
| DGI                                          |    Attributed    |  Contrastive/L-G  |                          Arbitrary                           |                     JS Estimator                      |        URL        | 2019 |
| HDMI                                         |    Attributed    |  Contrastive/L-G  |                     Attribute Shuffling                      |                     JS Estimator                      |        URL        | 2021 |
| DMGI                                         |  Heterogeneous   |  Contrastive/L-G  |                     Attribute Shuffling                      |                 JS Estimator<br/>MAE                  |        URL        | 2020 |
| MVGRL                                        |    Attributed    |  Contrastive/L-G  | Attribute Masking<br/>Edge Perturbation<br/>Edge Diffusion<br/>Random Walk Sampling | DV Estimator<br/>JS Estimator<br/>NT-Xent<br/>InfoNCE |        URL        | 2020 |
| HDGI                                         |  Heterogeneous   |  Contrastive/L-G  |                     Attribute Shuffling                      |                     JS Estimator                      |        URL        | 2019 |
| Subg-Con                                     |    Attributed    |  Contrastive/L-C  |                     Importance Sampling                      |                    Triplet Margin                     |        URL        | 2020 |
| Cotext Prediction                            |    Attributed    |  Contrastive/L-C  |                      Ego-nets Sampling                       |                          CE                           |       P\&F        | 2019 |
| GIC                                          |    Attributed    |  Contrastive/L-C  |                          Arbitrary                           |                     JS Estimator                      |        URL        | 2020 |
| GraphLoG                                     |    Attributed    |  Contrastive/L-C  |                      Attribute Masking                       |                        InfoNCE                        |        URL        | 2021 |
| MHCN                                         |  Heterogeneous   |  Contrastive/L-C  |                     Attribute Shuffling                      |                        InfoNCE                        |        JL         | 2021 |
| EGI                                          |    Attributed    |  Contrastive/L-C  |                      Ego-nets Sampling                       |                     SP Estimator                      |       P\&F        | 2020 |
| MICRO-Graph                                  |    Attributed    |  Contrastive/C-G  |                      Knowledge Sampling                      |                        InfoNCE                        |        URL        | 2020 |
| InfoGraph                                    |    Attributed    |  Contrastive/C-G  |                             None                             |                     SP Estimator                      |        URL        | 2019 |
| SUGAR                                        |    Attributed    |  Contrastive/C-G  |                         BFS Sampling                         |                     JS Estimator                      |        JL         | 2021 |
| BiGI                                         |  Heterogeneous   |  Contrastive/C-G  |           Edge Perturbation<br/>Ego-nets Sampling            |                     JS Estimator                      |        JL         | 2021 |
| HTC                                          |    Attributed    |  Contrastive/C-G  |                     Attribute Shuffling                      |             SP Estimator<br/>DV Estimator             |        URL        | 2021 |
| Node Property Prediction                     |    Attributed    |   Predictive/NP   |                             None                             |                          MAE                          |      P\&F/JL      | 2020 |
| S2GRL                                        |    Attributed    |   Predictive/CP   |                             None                             |                          CE                           |        URL        | 2020 |
| PairwiseDistance                             |    Attributed    |   Predictive/CP   |                             None                             |                          CE                           |      P\&F/JL      | 2020 |
| PairwiseAttrSim                              |    Attributed    |   Predictive/CP   |                             None                             |                          MAE                          |      P\&F/JL      | 2020 |
| Distance2Cluster                             |    Attributed    |   Predictive/CP   |                             None                             |                          MAE                          |      P\&F/JL      | 2020 |
| EdgeMask                                     |    Attributed    |   Predictive/CP   |                             None                             |                          CE                           |      P\&F/JL      | 2020 |
| TopoTER                                      |    Attributed    |   Predictive/CP   |                      Edge Perturbation                       |                          CE                           |        URL        | 2021 |
| Centrality Score Ranking                     |    Attributed    |   Predictive/CP   |                             None                             |                          CE                           |       P\&F        | 2019 |
| Meta-path prediction                         |  Heterogeneous   |   Predictive/CP   |                             None                             |                          CE                           |        JL         | 2020 |
| SLiCE                                        |  Heterogeneous   |   Predictive/CP   |                             None                             |                          CE                           |       P\&F        | 2020 |
| Distance2Labeled                             |    Attributed    |   Predictive/CP   |                             None                             |                          MAE                          |      P\&F/JL      | 2020 |
| ContextLabel                                 |    Attributed    |   Predictive/CP   |                             None                             |                          MAE                          |      P\&F/JL      | 2020 |
| HCM                                          |    Attributed    |   Predictive/CP   |                      Edge Perturbation                       |                  Bayesian inference                   |        URL        | 2021 |
| Contextual Molecular<br/>Property Prediction |    Attributed    |   Predictive/DK   |                             None                             |                          CE                           |       P\&F        | 2020 |
| Graph-level Motif Prediction                 |    Attributed    |   Predictive/DK   |                             None                             |                          CE                           |       P\&F        | 2020 |
| Multi-stage Self-training                    |    Attributed    |   Predictive/ST   |                             None                             |                         None                          |        JL         | 2018 |
| Node Clustering                              |    Attributed    |   Predictive/ST   |                             None                             |                      Clustering                       |      P\&F/JL      | 2020 |
| Graph Partitioning                           |    Attributed    |   Predictive/ST   |                             None                             |                     Partitioning                      |      P\&F/JL      | 2020 |
| CAGAN                                        |    Attributed    |   Predictive/ST   |                             None                             |                      Clustering                       |        URL        | 2020 |
| M3S################                          |    Attributed    | ##Predictive/ST## |                     ########None########                     |                  ####Clustering####                   |        JL         | 2020 |
| Cluster Preserving                           |    Attributed    |   Predictive/ST   |                             None                             |                   Clustering<br/>CE                   |       P\&F        | 2019 |

## A Summary of Implementation Details

About Task Level, Evaluation Metric, and Evaluation Datasets.

| Methods                                  |   Task Level    | Evaluation Metric                                            |                           Dataset                            |
| :--------------------------------------- | :-------------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| Graph Completion                         |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Node Attribute Masking                   |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| Edge Attribute Masking                   |      Graph      | Graph Classification (ROC-AUC)                               | MUTAG, PTC, PPI, BBBP, Tox21, ToxCast, ClinTox, MUV, HIV, SIDER, BACE |
| Node Attribute and Embedding Denoising   |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Adjacency Matrix Reconstruction          |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Graph Bert                               |      Node       | Node Classification (Acc)<br/>Node Clustering (NMI)          |                    Cora, Citeseer, Pubmed                    |
| Pretrain-Recsys                          |    Node/Link    | -                                                            |                   ML-1M, MOOCs and Last-FM                   |
| GPT-GNN                                  |    Node/Link    | Node Classification (F1-score)<br/>Link Prediction (ROC-AUC) |                     OAG, Amazon, Reddit                      |
| GraphCL                                  |      Graph      | Graph Classification  (Acc, ROC-AUC)                         | NCI1, PROTEINS, D\&D, COLLAB, RDT-B, RDT-M5K, GITHUB, MNIST, CIFAR10, MUTAG, IMDB-B, BBBP, Tox21, ToxCast, SIDER, ClinTox, MUV, HIV, BACE, PPI |
| IGSD                                     |      Graph      | Graph Classification (Acc)                                   |      MUTAG, PTC\_MR, NCI1, IMDB-B, QM9, COLLAB, IMDB-M       |
| DACL                                     |      Graph      | Graph Classification (Acc)                                   |        MUTAG, PTC\_MR, IMDB-B, IMDB-M, RDT-B, RDT-M5K        |
| LCL                                      |      Graph      | Graph Classification (Acc)                                   |   IMDB-B, IMDB-M, COLLAB, MUTAG, PROTEINS, PTC, NCI1, D\&D   |
| CSSL                                     |      Graph      | Graph Classification (Acc)                                   |          PROTEINS, D\&D, NCI1, NCI109, Mutagenicity          |
| GCC                                      |   Node/Graph    | Node Classification (Acc)<br/>Graph Classification (Acc)     | US-Airport, H-index, COLLAB, IMDB-B, IMDB-M, RDT-B, RDT-M5K  |
| GRACE                                    |      Node       | Node Classification (Acc, Micro-F1)                          |          Cora, Citeseer, Pubmed, DBLP, Reddit, PPI           |
| GCA                                      |      Node       | Node Classification (Acc)                                    | Wiki-CS, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics |
| GROC                                     |      Node       | Node Classification (Acc)                                    |        Cora, Citeseer, Pubmed, Amazon-Photo, Wiki-CS         |
| STDGI                                    |      Node       | Node Regression (MAE, RMSE, MAPE)                            |                           METR-LA                            |
| GMI                                      |    Node/Link    | Node Classification (Acc, Micro-F1)<br/>Link Prediction (ROC-AUC) |   Cora, Citeseer, PubMed, Reddit, PPI, BlogCatalog, Flickr   |
| KS2L                                     |    Node/Link    | Node Classification (Acc)<br/>Link Prediction (ROC-AUC)      | Cora, Citeseer, Pubmed, Amazon-Computers, Amazon-Photo, Coauthor-CS |
| CG3                                      |      Node       | Node Classification (Acc)                                    | Cora, Citeseer, Pubmed, Amazon-Computers, Amazon-Photo, Coauthor-CS |
| BGRL                                     |      Node       | Node Classification (Acc, Micro-F1)                          | Wiki-CS, Amazon-Computers, Amazon-Photo, PPI, Coauthor-CS, Coauthor-Physics, ogbn-arxiv |
| SelfGNN                                  |      Node       | Node Classification (Acc)                                    | Cora, Citeseer, Pubmed, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics |
| PT-DGNN                                  |      Link       | Link Prediction (ROC-AUC)                                    |               HepPh, Math Overflow, Super User               |
| COAD                                     |    Node/Link    | Node Clustering<br/>(Precision, Recall, F1-score)<br/>Link Prediction (HitRatio@K, MRR) |                    AMiner, News, LinkedIn                    |
| Contrast-Reg                             |    Node/Link    | Node Classification (Acc)<br/>Node Clustering<br/>(NMI, Acc, Macro-F1)<br/>Link Prediction (ROC-AUC) | Cora, Citeseer, Pubmed, Reddit, ogbn-arxiv,  Wikipedia, ogbn-products, Amazo-Computers, Amazo-Photo |
| DGI                                      |      Node       | Node Classification (Acc, Micro-F1)                          |             Cora, Citeseer, Pubmed, Reddit, PPI              |
| HDMI                                     |      Node       | Node Classification<br/>(Micro-F1, Macro-F1)<br/>Node Clustering (NMI) |                   ACM, IMDB, DBLP, Amazon                    |
| DMGI                                     |      Node       | Node Clustering (NMI)<br/>Node Classification (Acc)          |                   ACM, IMDB, DBLP, Amazon                    |
| MVGRL                                    |   Node/Graph    | Node Classification (Acc)<br/>Node Clustering (NMI, ARI)<br/>Graph Classification (Acc) | Cora, Citeseer, Pubmed, MUTAG, PTC\_MR, IMDB-B, IMDB-M, RDT-B |
| HDGI                                     |      Node       | Node Classification<br/>(Micro-F1, Macro-F1)<br/>Node Clustering (NMI, ARI) |                       ACM, DBLP, IMDB                        |
| Subg-Con                                 |      Node       | Node Classification (Acc, Micro-F1)                          |         Cora, Citeseer, Pubmed, PPI, Flickr, Reddit          |
| Cotext Prediction                        |      Graph      | Graph Classification (ROC-AUC)                               | MUTAG, PTC, PPI, BBBP, Tox21, ToxCast, ClinTox, MUV, HIV, SIDER, BACE |
| GIC                                      |    Node/Link    | Node Classification (Acc)<br/>Node Clustering (Acc, NMI, ARI)<br/>Link Prediction (ROC-AUC, ROC-AP) | Cora, Citeseer, Pubmed, Amazon-Computers, Amazon-Photo, Coauthor-CS, Coauthor-Physics |
| GraphLoG                                 |      Graph      | Graph Classification (ROC-AUC)                               |     BBBP, Tox21, ToxCast, ClinTox, MUV, HIV, SIDER, BACE     |
| MHCN                                     |    Node/Link    | -                                                            |                    Last-FM, Douban, Yelp                     |
| EGI                                      |    Node/Link    | Node Classification (Acc)<br/>Link Prediction (ROC-AUC, MRR) |                        YAGO, Airport                         |
| MICRO-Graph                              |      Graph      | Graph Classification (ROC-AUC)                               |       BBBP, Tox21, ToxCast, ClinTox, HIV, SIDER, BACE        |
| InfoGraph                                |      Graph      | Graph Classification (Acc)                                   |     MUTAG, PTC\_MR, RDT-B, RDT-M5K, IMDB-B, QM9, IMDB-M      |
| SUGAR                                    |      Graph      | Graph Classification (Acc)                                   |           MUTAG, PTC, PROTEINS, D\&D, NCI1, NCI109           |
| BiGI                                     |      Link       | Link Prediction (AUC-ROC, AUC-PR)                            |               DBLP, ML-100K, ML-1M, Wikipedia                |
| HTC                                      |      Graph      | Graph Classification (Acc)                                   |     MUTAG, PTC\_MR, IMDB-B, IMDB-M, RDT-B, QM9, RDT-M5K      |
| Node Property Prediction                 |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| S2GRL                                    |    Node/Link    | Node Classification (Acc, Micro-F1)<br/>Node Clustering (NMI)<br/>Link Prediction (ROC-AUC) |   Cora, Citeseer, Pubmed, PPI, Flickr, BlogCatalog, Reddit   |
| PairwiseDistance                         |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| PairwiseAttrSim                          |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| Distance2Cluster                         |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| EdgeMask                                 |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| TopoTER                                  |   Node/Graph    | Node Classification (Acc)<br/>Graph Classification (Acc)     | Cora, Citeseer, Pubmed, MUTAG, PTC-MR, RDT-B, RDT-M5K, IMDB-B, IMDB-M |
| Centrality Score Ranking                 | Node/Link/Graph | Node Classification (Micro-F1)<br/>Link Prediction (Micro-F1)<br/>Graph Classification (Micro-F1) |         Cora, Pubmed, ML-100K, ML-1M, IMDB-M, IMDB-B         |
| Meta-path prediction                     |    Node/Link    | Node Classification (F1-score)<br/>Link Prediction (ROC-AUC) |              ACM, IMDB, Last-FM, Book-Crossing               |
| SLiCE                                    |      Link       | Link Prediction (ROC-AUC, Micro-F1)                          |         Amazon, DBLP, Freebase, Twitter, Healthcare          |
| Distance2Labeled                         |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| ContextLabel                             |      Node       | Node Classification (Acc)                                    |                Cora, Citeseer, Pubmed, Reddit                |
| HCM                                      |      Node       | Node Classification (ROC-AUC)                                |           ACM, Amazon, Enron, BlogCatalog, Flickr            |
| Contextual Molecular Property Prediction |      Graph      | Graph Classification (Acc)<br/>Graph Regression (MAE)        | BBBP, SIDER, ClinTox, BACE, Tox21, ToxCast, ESOL, FreeSolv, Lipo, QM7, QM8 |
| Graph-level Motif Prediction             |      Graph      | Graph Classification (Acc)<br/>Graph Regression (MAE)        | BBBP, SIDER, ClinTox, BACE, Tox21, ToxCast, ESOL, FreeSolv, Lipo, QM7, QM8 |
| Multi-stage Self-training                |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Node Clustering                          |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Graph Partitioning                       |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| CAGAN                                    |      Node       | Node Classfication<br/>(Micro-F1, Macro-F1)<br/>Node Clustering<br/>(Micro-F1, Macro-F1, NMI) |                    Cora, Citeseer, Pubmed                    |
| M3S                                      |      Node       | Node Classification (Acc)                                    |                    Cora, Citeseer, Pubmed                    |
| Cluster Preserving                       | Node/Link/Graph | Node Classification (Micro-F1)<br/>Link Prediction (Micro-F1)<br/>Graph Classification (Micro-F1) |         Cora, Pubmed, ML-100K, ML-1M, IMDB-M, IMDB-B         |

## A summary of Common Graph Datasets

About category, graph number, node number per graph, edge number per graph, dimensionality of node attributes, class number, and citation papers.

|     Dataset      |     Category      | \#Graph | \#Node (Avg.) | \#Edge (Avg.) | \#Feature | \#Class |
| :--------------: | :---------------: | :-----: | :-----------: | :-----------: | :-------: | :-----: |
|       Cora       | Citation Network  |    1    |     2708      |     5429      |   1433    |    7    |
|     Citeseer     | Citation Network  |    1    |     3327      |     4732      |   3703    |    6    |
|      Pubmed      | Citation Network  |    1    |     19717     |     44338     |    500    |    3    |
|     Wiki-CS      | Citation Network  |    1    |     11701     |    216123     |    300    |   10    |
|   Coauthor-CS    | Citation Network  |    1    |     18333     |     81894     |   6805    |   15    |
| Coauthor-Physics | Citation Network  |    1    |     34493     |    247962     |   8415    |    5    |
|    DBLP (v12)    | Citation Network  |    1    |    4894081    |   45564149    |     -     |    -    |
|    ogbn-arxiv    | Citation Network  |    1    |    169343     |    1166243    |    128    |   40    |
|      Reddit      |  Social Network   |    1    |    232965     |   11606919    |    602    |   41    |
|   BlogCatalog    |  Social Network   |    1    |     5196      |    171743     |   8189    |    6    |
|      Flickr      |  Social Network   |    1    |     7575      |    239738     |   12047   |    9    |
|      COLLAB      |  Social Networks  |  5000   |     74.49     |    2457.78    |     -     |    2    |
|      RDT-B       |  Social Networks  |  2000   |    429.63     |    497.75     |     -     |    2    |
|     RDT-M5K      |  Social Networks  |  4999   |    508.52     |    594.87     |     -     |    5    |
|      IMDB-B      |  Social Networks  |  1000   |     19.77     |     96.53     |     -     |    2    |
|      IMDB-M      |  Social Networks  |  1500   |     13.00     |     65.94     |     -     |    3    |
|     ML-100K      |  Social Networks  |    1    |     2625      |    100000     |     -     |    5    |
|      ML-1M       |  Social Networks  |    1    |     9940      |    1000209    |     -     |    5    |
|       PPI        | Protein Networks  |   24    |     56944     |    818716     |    50     |   121   |
|       D\&D       | Protein Networks  |  1178   |    284.32     |    715.65     |    82     |    2    |
|     PROTEINS     | Protein Networks  |  1113   |     39.06     |     72.81     |     4     |    2    |
|       NCI1       |  Molecule Graphs  |  4110   |     29.87     |     32.30     |    37     |    2    |
|      MUTAG       |  Molecule Graphs  |   188   |     17.93     |     19.79     |     7     |    2    |
|       PTC        |  Molecule Graphs  |   344   |     25.50     |       -       |    19     |    2    |
|  QM9 (QM7, QM8)  |  Molecule Graphs  | 133885  |       -       |       -       |     -     |    -    |
|       BBBP       |  Molecule Graphs  |  2039   |     24.05     |     25.94     |     -     |    2    |
|      Tox21       |  Molecule Graphs  |  7831   |     18.51     |     25.94     |     -     |   12    |
|     ToxCast      |  Molecule Graphs  |  8575   |     18.78     |     19.26     |     -     |   167   |
|     ClinTox      |  Molecule Graphs  |  1478   |     26.13     |     27.86     |     -     |    2    |
|       MUV        |  Molecule Graphs  |  93087  |     24.23     |     26.28     |     -     |   17    |
|       HIV        |  Molecule Graphs  |  41127  |     25.53     |     27.48     |     -     |    2    |
|      SIDER       |  Molecule Graphs  |  1427   |     33.64     |     35.36     |     -     |   27    |
|       BACE       |  Molecule Graphs  |  1513   |     34.12     |     36.89     |     -     |    2    |
|     PTC\_MR      |  Molecule Graphs  |   344   |     14.29     |     14.69     |     -     |    2    |
|      NCI109      |  Molecule Graphs  |  4127   |     29.68     |     32.13     |     -     |    2    |
|   Mutagenicity   |  Molecule Graphs  |  4337   |     30.32     |     30.77     |     -     |    2    |
|      MNIST       |  Others (Image)   |    -    |     70000     |       -       |    784    |   10    |
|     CIFAR10      |  Others (Image)   |    -    |     60000     |       -       |   1024    |   10    |
|     METR-LA      | Others (Traffic)  |    1    |      207      |     1515      |     2     |    -    |
| Amazon-Computers | Others (Purchase) |    1    |     13752     |    245861     |    767    |   10    |
|   Amazon-Photo   | Others (Purchase) |    1    |     7650      |    119081     |    745    |    8    |
|  ogbn-products   | Others (Purchase) |    1    |    2449029    |   61859140    |    100    |   47    |

## A summary of Open-source Codes (Github)

| Methods                                  | Github                                                       |
| :--------------------------------------- | :----------------------------------------------------------- |
| Graph Completion                         | https://github.com/Shen-Lab/SS-GCNs                          |
| Node Attribute Masking                   | https://github.com/ChandlerBang/SelfTask-GNN                 |
| Edge Attribute Masking                   | http://snap.stanford.edu/gnn-pretrain                        |
| Attribute and Embedding Denoising        | N.A.                                                         |
| Adjacency Matrix Reconstruction          | N.A.                                                         |
| Graph Bert                               | https://github.com/anonymous-sourcecode/Graph-Bert           |
| Pretrain-Recsys                          | https://github.com/jerryhao66/Pretrain-Recsys                |
| GPT-GNN                                  | https://github.com/acbull/GPT-GNN                            |
| GraphCL                                  | https://github.com/Shen-Lab/GraphCL                          |
| IGSD                                     | N.A.                                                         |
| DACL                                     | N.A.                                                         |
| LCL                                      | https://github.com/YuxiangRen                                |
| CSSL                                     | N.A.                                                         |
| GCC                                      | https://github.com/THUDM/GCC                                 |
| GRACE                                    | https://github.com/CRIPAC-DIG/GRACE                          |
| GCA                                      | https://github.com/CRIPAC-DIG/GCA                            |
| GROC                                     | N.A.                                                         |
| STDGI                                    | N.A.                                                         |
| GMI                                      | https://github.com/zpeng27/GMI                               |
| KS2L                                     | N.A.                                                         |
| CG3                                      | N.A.                                                         |
| BGRL                                     | N.A.                                                         |
| SelfGNN                                  | https://github.com/zekarias-tilahun/SelfGNN                  |
| PT-DGNN                                  | https://github.com/Mobzhang/PT-DGNN                          |
| COAD                                     | https://github.com/allanchen95/Expert-Linking                |
| Contrast-Reg                             | N.A.                                                         |
| DGI                                      | https://github.com/PetarV-/DGI                               |
| HDMI                                     | N.A.                                                         |
| DMGI                                     | https://github.com/pcy1302/DMGI                              |
| MVGRL                                    | https://github.com/kavehhassani/mvgrl                        |
| HIGI                                     | https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax |
| Subg-Con                                 | https://github.com/yzjiao/Subg-Con                           |
| Cotext Prediction                        | http://snap.stanford.edu/gnn-pretrain                        |
| GIC                                      | https://github.com/cmavro/Graph-InfoClust-GIC                |
| GraphLoG                                 | https://openreview.net/forum?id=DAaaaqPv9-q                  |
| MHCN                                     | https://github.com/Coder-Yu/RecQ                             |
| EGI                                      | https://openreview.net/forum?id=J_pvI6ap5Mn                  |
| MICRO-Graph                              | https://drive.google.com/file/d/1b751rpnV-SDmUJvKZZI-AvpfEa9eHxo9/ |
| InfoGraph                                | https://github.com/fanyun-sun/InfoGraph                      |
| SUGAR                                    | https://github.com/RingBDStack/SUGAR                         |
| BiGI                                     | https://github.com/clhchtcjj/BiNE                            |
| HTC                                      | N.A.                                                         |
| Node Property Prediction                 | https://github.com/ChandlerBang/SelfTask-GNN                 |
| S2GRL                                    | N.A.                                                         |
| PairwiseDistance                         | https://github.com/ChandlerBang/SelfTask-GNN                 |
| PairwiseAttrSim                          | https://github.com/ChandlerBang/SelfTask-GNN                 |
| Distance2Cluster                         | https://github.com/ChandlerBang/SelfTask-GNN                 |
| EdgeMask                                 | https://github.com/ChandlerBang/SelfTask-GNN                 |
| TopoTER                                  | N.A.                                                         |
| Centrality Score Ranking                 | N.A.                                                         |
| Meta-path prediction                     | https://github.com/mlvlab/SELAR                              |
| SLiCE                                    | https://github.com/pnnl/SLICE                                |
| Distance2Labeled                         | https://github.com/ChandlerBang/SelfTask-GNN                 |
| ContextLabel                             | https://github.com/ChandlerBang/SelfTask-GNN                 |
| HCM                                      | N.A.                                                         |
| Contextual Molecular Property Prediction | https://github.com/tencent-ailab/grover                      |
| Graph-level Motif Prediction             | https://github.com/tencent-ailab/grover                      |
| Multi-stage Self-training                | https://github.com/Davidham3/deeper_insights_into_GCNs       |
| Node Clustering                          | https://github.com/Shen-Lab/SS-GCNs                          |
| Graph Partitioning                       | https://github.com/Shen-Lab/SS-GCNs                          |
| CAGAN                                    | N.A.                                                         |
| M3S                                      | https://github.com/datake/M3S                                |
| Cluster Preserving                       | N.A.                                                         |

