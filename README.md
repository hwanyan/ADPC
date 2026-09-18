# ADPC: Automated Clustering with Density Peaks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)

A Python implementation of **ADPC**, a fully automated density-peak-based
clustering algorithm that requires **no manually specified
hyper-parameters** (no cutoff distance, no fixed neighborhood size, and no
human intervention for cluster-center selection). ADPC adaptively derives
its neighborhood scale from *natural neighbor* search, and automatically
estimates the number of clusters, detects cluster centers, and assigns
every sample to its most appropriate cluster.

## About This Repository

This repository is the **official source-code repository** accompanying
the paper *"Automated clustering with density peaks"*. The paper itself
is a collaborative work of five co-authors (see [Paper Authors](#paper-authors)
below); this repository, however, is **individually created and
maintained by Huan Yan** (one of the paper's co-authors, responsible for
the algorithm implementation). Huan Yan is the maintainer in charge of
the code releases, issue tracking, and pull requests for this repository,
while the scientific contribution and authorship of the underlying
research remain jointly credited to all paper co-authors as listed below.

## Abstract

> Clustering is a significant problem in machine learning and data mining
> fields. However, automated clustering has been an open problem. DPC
> (Density Peak Clustering) algorithm opened the door to this problem, but
> it and its variants require some parameters to be provided in advance.
> Moreover, DPC requires human intervention to select cluster centers,
> thereby introducing randomness to the final clustering outcomes. To
> address these limitations, an automated clustering algorithm with
> density peaks, referred to ADPC, is proposed in this paper to
> automatically identify cluster centers and clusters within a dataset.
> This ADPC comprises three stages: 1) Estimate the number of clusters in
> a dataset; 2) Identify cluster centers of the dataset; 3) Assign points
> to most appropriate clusters. The most significant contribution of ADPC
> is its capability to detect clusters of a dataset automatically without
> human interventions by estimating the number of clusters within a
> dataset and complete the clustering process without any prior
> knowledge. Experiments on both synthetic and real-world datasets
> demonstrate the power of ADPC.

## Highlights

- Proposing an automatic way to estimate the number of clusters in a dataset.
- Introducing an automatic cluster center detection method.
- Presenting an automatic assignment strategy for assigning points to clusters.
- Developing an automated clustering algorithm for detecting clusters in a dataset.

## Algorithm Pipeline

ADPC completes clustering through the following stages (see the
module-level and per-method docstrings in [`ADPC.py`](./ADPC.py) for full
algorithmic details):

1. **Distance matrix construction** — min-max feature normalization
   followed by pairwise Euclidean distance computation.
2. **Natural neighbor search** — adaptively determines the neighborhood
   size `k` and the search radius `lamda` without any manual input.
3. **Local density & relative distance estimation** for every sample.
4. **Local density peak (LDP) detection** and representative-point
   assignment, followed by representative-chain adjustment.
5. **Automatic cluster-number estimation** via shared-nearest-neighbor
   weighted hierarchical clustering of the local density peaks.
6. **Cluster-center selection** and initial label assignment.
7. **Fuzzy-membership-based label propagation** for samples left
   unassigned after the initial round.
8. **Final label completion** for any residual isolated samples.

## Requirements

- Python >= 3.7
- numpy
- scikit-learn
- scipy

Install the dependencies:

```bash
pip install numpy scikit-learn scipy
```

## Usage

```python
import numpy as np
from ADPC import ADPC

# X: feature matrix of shape (n_samples, n_features)
X = np.array([...])

model = ADPC(X)
model.fit()

print("Predicted labels:", model.Labels)          # cluster label per sample
print("Cluster centers:", model.ClusterCenters)   # indices of cluster centers
print("Estimated cluster number:", model.CN)
```

## Citation

If you use this code in your research, please cite the corresponding paper:

> Xie, J., Yan, H., Wang, M., Grant, P. W., & Pedrycz, W. (2026).
> Automated clustering with density peaks. *Manuscript submitted to
> Pattern Recognition.*

BibTeX:

```bibtex
@article{yan2026adpc,
    title   = {Automated clustering with density peaks},
    author  = {Xie, Juanying and Yan, Huan and Wang, Mingzhao and
                Grant, Philip W. and Pedrycz, Witold},
    journal = {Pattern Recognition},
    year    = {2026},
    note    = {Manuscript submitted for publication}
}
```

*This citation will be updated with the final publication details
(volume, pages, DOI) once the paper is officially accepted and published.*

## Paper Authors

The paper *"Automated clustering with density peaks"* is a joint effort
of the following co-authors (corresponding author marked with \*):

| Author | Affiliation |
|---|---|
| Juanying Xie\* | School of Computer Science, Shaanxi Normal University, Xi'an, China; School of Mathematics and Statistics, Xi'an Jiaotong University, Xi'an, China |
| **Huan Yan** | School of Computer Science, Shaanxi Normal University, Xi'an, China |
| Mingzhao Wang | School of Computer Science, Shaanxi Normal University, Xi'an, China |
| Philip W. Grant | Department of Computer Science, Swansea University, Swansea, UK |
| Witold Pedrycz | Department of Electrical and Computer Engineering, University of Alberta, Edmonton, Canada |

> Note: This repository only reflects the maintenance responsibility of
> Huan Yan for the released code. Scientific credit for the research
> presented in the paper is shared jointly among all co-authors above.

## Repository Maintainer

| | |
|---|---|
| **Name (中文)** | 严欢 |
| **Name (English)** | Huan Yan (hwanyan) |
| **Email** | yan-huan@snnu.edu.cn |
| **ORCID** | [https://orcid.org/0009-0000-0016-6324](https://orcid.org/0009-0000-0016-6324) |
| **Affiliations** | 1. Tencent Cloud Computing (Chongqing) Co., Ltd. <br> 2. Shaanxi Normal University |
| **Role** | Paper co-author; sole creator and maintainer of this official code repository |

## License

This project is released under the [MIT License](./LICENSE).
