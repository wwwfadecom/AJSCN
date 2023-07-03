# AJSCN
Adversarially Regularized Joint Structured Clustering Network


# Dataset
Due to file size limitations, we only put USPS and HHAR datasets in the project. 
All datasets are public datasets, which can be found through related literature


# Code
```
python AJSCN.py --name [usps|hhar|reut|acm|dblp|wiki]
```
# How to run AJSCN on other datasets?
1.For non-graph data, calculate the KNN graph by calcu_graph.py. For graph data, you only need to put the original graph in the graph folder.
2.Pre-train the autoencoder, see file data/pretrain.py for details.
3.For graph data and non-graph data, make some modifications to the parameters according to the description of the paper and then run.

