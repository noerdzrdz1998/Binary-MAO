# Binary Mexican Axolotl Optimization (BMAO)

**A Bio-Inspired Algorithm for Feature Selection in Machine Learning**

[![Conference](https://img.shields.io/badge/MLPR-2025-blue)](https://doi.org/10.1145/3760678.3760693)  
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview
This repository contains the reference implementation of **Binary Mexican Axolotl Optimization (BMAO)**, a metaheuristic designed for **feature selection** in supervised learning.  

BMAO extends the Mexican Axolotl Optimization algorithm to a binary search space, modeling biological processes of the axolotl—**reproduction, injury, and regeneration**—to balance exploration and exploitation when searching for compact and predictive subsets of features.  

The algorithm has been evaluated on a wide range of **UCI-style datasets**, showing improvements in classification accuracy and dimensionality reduction compared to using all features.  

This work was presented at the **3rd International Conference on Machine Learning and Pattern Recognition (MLPR 2025, Kyoto, Japan)** and published in the **ACM ICPS** proceedings.  
📄 DOI: [10.1145/3760678.3760693](https://doi.org/10.1145/3760678.3760693)

---

## Repository Structure
```
Binary-MAO-main/
│── feature_selection_MAO.py    # BMAO pipeline (cross-validation + evaluation)
│── mao_binary.py               # Core binary MAO implementation
│── data_loader.py              # Dataset loader and preprocessing
│── requirements.txt            # Dependencies
│── all_datasets/               # Benchmark datasets (.dat, .csv, .xlsx)
```
> Datasets are expected to have the **class label in the last column**.  
> Non-numeric labels are automatically mapped to integers for binary classification.

---

## Installation
```bash
git clone https://github.com/<your-username>/Binary-MAO.git
cd Binary-MAO-main
pip install -r Binary-MAO-main/requirements.txt
```

Python ≥3.8 recommended.

---

## Example Usage
The pipeline can be called directly from Python with any scikit-learn compatible classifier and metric:

```python
import os, sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Add repo to path
sys.path.append("Binary-MAO-main")

from data_loader import process_files
from feature_selection_MAO import model_with_metaheuristic_feature_selection

# Load datasets
dfs, names = process_files("Binary-MAO-main/all_datasets")

# Select one dataset, e.g. Pima
df = dfs[names.index("pima")]

# Run BMAO
results = model_with_metaheuristic_feature_selection(
    datasets=[df],
    datasets_names=["pima"],
    model=lambda: SVC(kernel="rbf", gamma="scale"),
    mao_metric="alpha",                   # "alpha", "alpha-mean" or custom callable
    evaluation_metric=accuracy_score,     # any sklearn metric
    validation_method="stratified_kfold",
    validation_params={"n_splits": 5, "shuffle": True, "random_state": 42},
    pop_size=30,
    max_iter=50,
    early_stopping_steps=10
)

print(results["pima"])
# {'mean_metric': <cv_score>, 'selected_features': [..], 'processing_time': <seconds>}
```

The function returns a dictionary per dataset with:
- `mean_metric` → performance score (e.g. accuracy)  
- `selected_features` → indices of chosen features  
- `processing_time` → runtime in seconds  

---
## Citation
If you use this work, please cite:

```
@inproceedings{alarcon2025bmao,
  title={Binary Mexican Axolotl Optimization (BMAO): A Bio-Inspired Algorithm for Enhancing Machine Learning Performance through Feature Selection},
  author={Rodríguez, Noé Oswaldo and Rosas, Carolina and Alarcón, Antonio and Yáñez-Márquez, Cornelio and Villuendas-Rey, Yenny and Recio García, Juan Antonio},
  booktitle={Proceedings of the 3rd International Conference on Machine Learning and Pattern Recognition (MLPR 2025)},
  year={2025},
  doi={10.1145/3760678.3760693}
}
```

---

## Authors
- Noé Oswaldo Rodríguez Rodríguez (CIC-IPN)  
- Carolina Rosas Alatriste (CIC-IPN)  
- **Antonio Alarcón Paredes (CIC-IPN)**  
- Cornelio Yáñez Márquez (CIC-IPN)  
- Yenny Villuendas-Rey (CIDETEC-IPN)  
- Juan Antonio Recio García (UCM Madrid)  

---

## License
This project is licensed under **CC BY 4.0**.  
See the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

