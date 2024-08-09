# DeepSignalingFlow

## 1. Model Overall Architecture
![](./figures/Figure1.png)

## 2. Run the DeepSignalingFlow
### 2.1 Parse the Datasets
```
python load_data.py
```

### 2.2 Run the GNN models for comparisons
```
python geo_tmain_gat.py
python geo_tmain_gcn.py
python geo_tmain_gformer.py
python geo_tmain_gin.py
python geo_tmain_mixhop.py
```
* Following one is DeepSignalingFlow
```
python geo_tmain_webgnn.py 
```

## 3. Model Results
![](./figures/Figure2.png)

## 4. Cell-line Specific Biomarkers
![](./figures/Figure3.png)

![](./figures/Figure4.png)

## 5. Result Validation
* Comparisons for Summations of Weights on Shortest Paths Smaller than 4 Between 2 Genes Targeted by Drug Combinations Ranking Top 10 and Bottom 10 in the NCI ALMANAC dataset.
![](./figures/Figure5.png)

* Comparisons for Summations of Weights on Shortest Paths Smaller than 5 Between 2 Genes Targeted by Drug Combinations Ranking Top 10 and Bottom 10 in the NCI ALMANAC dataset.
![](./figures/Figure6.png)
