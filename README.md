# DeepSignalingFlow: an integrative and interpretable graph AI framework to uncover mechanisms of synergy (MoS) in drug combinations by mining multi-omic data

<p align="center">
  <a href="https://www.nature.com/articles/s41540-024-00421-w">
    <img src="https://img.shields.io/badge/Paper-npj%20Systems%20Biology%20%26%20Applications-1f8ecd" alt="npj Systems Biology & Applications (Nature) Paper">
  </a>
  <a href="https://github.com/FuhaiLiAiLab/DeepSignalingFlow">
    <img src="https://img.shields.io/badge/GitHub-DeepSignalingFlow-181717?logo=github" alt="GitHub Repo">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-red" alt="MIT License">
  </a>
</p>

---

Complex signaling pathways are often implicated in drug resistance. Combining drugs to perturb multiple signaling targets can mitigate resistance, but understanding **mechanisms of synergy (MoS)** remains challenging. Large-scale multi-omic resources and experimentally measured drug-combination synergy scores (e.g., **NCI ALMANAC**, **O’Neil**, **DrugComb**, **DrugCombDB**) create new opportunities to study MoS systematically. **DeepSignalingFlow** models the **directed signaling flow** from disease-critical proteins (with multi-omic evidence) to candidate drug targets—an approach not used by prior methods. Across four synergy datasets, DeepSignalingFlow achieves **state-of-the-art prediction** and interprets MoS via **core signaling flows**, paving the way for precision combination therapy.

---

## Contents

- [1. Model Architecture](#1-model-architecture)
- [2. Run DeepSignalingFlow](#2-run-deepsignalingflow)
  - [2.1 Parse the datasets](#21-parse-the-datasets)
  - [2.2 Train baselines](#22-train-baselines)
  - [2.3 Train DeepSignalingFlow](#23-train-deepsignalingflow)
- [3. Results](#3-results)
- [4. Cell-line Specific Biomarkers](#4-cell-line-specific-biomarkers)
- [5. Result Validation](#5-result-validation)
- [Links](#links)
- [Citation](#citation)
- [License](#license)

---

## 1. Model Architecture

![](./figures/Figure1.png)

---

## 2. Run DeepSignalingFlow

### 2.1 Parse the datasets

```bash
python load_data.py
```

### 2.2 Train baselines

```bash
python geo_tmain_gat.py
python geo_tmain_gcn.py
python geo_tmain_gformer.py
python geo_tmain_gin.py
python geo_tmain_mixhop.py
```

### 2.3 Train DeepSignalingFlow

```bash
python geo_tmain_webgnn.py
```

---

## 3. Results

<p align="center">
  <img src="./figures/Figure2.png" alt="Overall results summary" width="70%">
</p>

---

## 4. Cell-line–Specific Biomarkers

<p align="center">
  <img src="./figures/Figure3.png" alt="Biomarker analysis (panel 1)" width="70%">
</p>
<p align="center">
  <img src="./figures/Figure4.png" alt="Biomarker analysis (panel 2)" width="70%">
</p>

---

## 5. Result Validation

- **Shortest-path weight sums (< 4 hops)** between two drug-target genes (Top-10 vs Bottom-10 combinations, **NCI ALMANAC**):
  <p align="center">
    <img src="./figures/Figure5.png" alt="Validation: <4-hop path weight sums" width="70%">
  </p>

- **Shortest-path weight sums (< 5 hops)** between two drug-target genes (Top-10 vs Bottom-10 combinations, **NCI ALMANAC**):
  <p align="center">
    <img src="./figures/Figure6.png" alt="Validation: <5-hop path weight sums" width="70%">
  </p>


---

## Citation

If you use **DeepSignalingFlow** in your research, please cite:

```bibtex
@article{zhang2024using,
  title={Using DeepSignalingFlow to mine signaling flows interpreting mechanism of synergy of cocktails},
  author={Zhang, Heming and Chen, Yixin and Payne, Philip and Li, Fuhai},
  journal={npj Systems Biology and Applications},
  volume={10},
  number={1},
  pages={92},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

---

## License

MIT License. See `LICENSE` for details.
