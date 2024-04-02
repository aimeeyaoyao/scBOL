# scBOL

Introduction
-----
Here we propose a new, challenging yet realistic task called universal cell type identification for single-cell and spatial transcriptomics data. In this task, we aim to give semantic labels to target cells from known cell types and cluster labels to those from novel ones. To tackle this problem, we propose an end-to-end algorithm called scBOL from the perspective of bipartite prototype alignment. Comprehensive results on various benchmarks demonstrate the superiority of scBOL over other state-of-the-art cell type identification methods.

For scRNA-seq data, the input of scBOL model is the mixed reference data and target data, the rows represent the cells and the columns represent the genes. For spatial transcriptomic data, we also need the cell spatial coordinates for graph construction. We supply the data preprocessing, network architecture, algorithm running in the corresponding Python files. 

Architecture
-----

Requirement
-----
The version of Python environment and packages we used can be summarized as follows,

python environment >=3.6

torch >=1.10.2

torch-geometric 1.7.0

torch-sparse 0.6.8

torch-cluster 1.5.9

scanpy 1.4.6

scikit-learn 0.20.4

scipy 1.5.4

jgraph 0.2.1

tqdm 4.64.1

pandas 1.1.5

numpy 1.19.5

...

Please build the corresponding operation environment before running our codes. 

Quickstart
-----
We provide some explanatory descriptions for the codes, please see the specific code files. We supply different training codes for scRNA-seq data and spatial transcriptomic data, respectively. You can find them according to the code folders. Specifically, for scRNA-seq data, if you want to use scBOL in the intra-data setting, you can focus on the "bol_train_single.py" script and run it in your command lines with corresponding optional parameters. Similarly, if you want to use our method in the inter-data setting, you can focus on the "bol_train_cross.py" script. For spatial transcriptomic data, the main model file is "bol_train.py" and the startup script is "bol_run.py". You can also pass in different optional parameters to run our model. For each scenario, you should put the data file in the correct data folders, and then you can obtain the final results of our model. 

Data
-----
The scRNA-seq datasets we used can be downloaded in <a href="https://cblast.gao-lab.org/download">data1</a>, and the spatial transcriptomic datasets we tested can be found in <a href="https://crukci.shinyapps.io/SpatialMouseAtlas/">data2</a>. 

Reference
-----
Our paper is accepted by Briefings in Bioinformatics and the specific details will come soon. Please consider citing it.

Contributing
-----
Author email: zhaiyuyao@stu.pku.edu.cn. If you have any questions, please contact with me.

