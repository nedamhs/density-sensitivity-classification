# Density Sensitivity ML Pipeline

A machine learning pipeline for predicting density sensitivity in chemical reactions using molecular strucuture and Coulomb matrices.

## 🔬 Project Overview

This project implements a complete ML pipeline to predict whether chemical reactions are sensitive to changes in electron density.  
Density-sensitive reactions are those where energy errors are driven by inaccuracies in the electron density, while density-insensitive reactions are those where errors arise primarily from the approximate functional form.

The pipeline integrates physics-based molecular encoding with modern ML techniques:

- **Molecular Parsing** – Reads and parses molecular geometries from `.xyz` files.  
- **Coulomb Matrix Descriptors** – Represents each molecule as a rotation- and permutation-invariant matrix capturing interatomic electrostatic interactions.  
- **Reaction Matrices** – Constructs block-diagonal reaction matrices that account for stoichiometric coefficients of reactants and products.  
- **Spectral Feature Extraction** – Computes and sorts eigenvalues of each reaction matrix to obtain fixed-length, invariant feature vectors.  
- **Learning and Prediction** – Trains **Decision Tree**, **Random Forest** and **XGBoost** models for **binary classification** (density sensitive vs. insensitive).


## 📁 Project Structure

```
density_sensitivity-classification/
├── Descriptor1/
│ ├── Descriptor1_complete_features.npy             — feature matrix (reaction eigenvalues + meta data)
│ ├── Descriptor1_complete_targets.npy              — target labels for reactions ( density sensitivity)
│ ├── descriptor1_model.ipynb                       — model training and evaluation notebook
│ └── dimensionality_reduction.ipynb                — PCA/UMAP/tsne analysis
│
├── diagonalize_matrices.py                         — computes eigenvalues of reaction matrices
├── generate_cm.py                                  — functions to construct Coulomb matrices for molecular systems
├── pad_and_metadata.py                             — pads eigenvalue vectors and appends metadata
├── preprocess.py                                   — util functions for preprocessing such as combining matrices
├── main.py                                         — orchestrates full descriptor generation workflow
├── final_dict_allsets.pkl                          — dictionary containing coulomb matrices for all systems of the GMTKN55
├── requirements.txt                                — Python dependencies
└── README.md —            
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nedamhs/density-sensitivity-classification.git
cd density-sensitivity-classification

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# generates datasets used for ML training
python main.py

```
  
## 📈 Model Performance


The dataset exhibits a moderate class imbalance (~33% density-sensitive vs. ~67% density-insensitive reactions). Models were evaluated using metrics robust to imbalance, including balanced accuracy, recall, and precision.

### Test set performance of each model at its optimal K\* (number of eigenvalues used)

| Model         | K\* | Accuracy | Balanced Accuracy | ROC-AUC | Recall (Minority) | Precision (Minority) |
|---------------|:---:|:--------:|:-----------------:|:-------:|:-----------------:|:--------------------:|
| **XGBoost**       | 22  | **0.821** | **0.812** | **0.883** | 0.784 | **0.710** |
| **Random Forest** | 22  | 0.801 | 0.791 | 0.864 | 0.763 | 0.679 |
| **Decision Tree** | 24  | 0.808 | 0.806 | 0.825 | **0.804** | 0.678 |

---

## 📝 Data 

- GMTKN55 database
- SWARM dataset 

## 🙏 Acknowledgments

- Burke Group @ UCI
- Goerigk Research Group @ university of Melbourne
- Stephan Grimme's group @ university of Bonn

## Resources
- <https://hunterheidenreich.com/posts/molecular-descriptor-coulomb-matrix/#the-coulomb-matrix>
- <https://goerigk.chemistry.unimelb.edu.au/research/the-gmtkn55-database>

## Reference
Goerigk, L.; Hansen, A.; Bauer, C.; Ehrlich, S.; Najibi, A.; Grimme, S.  
*A look at the density functional theory zoo with the advanced GMTKN55 database for general main group thermochemistry, kinetics and noncovalent interactions.*  
**Phys. Chem. Chem. Phys.** 2017, 19, 32184–32215.  
DOI: [10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G)

Sim, E.; Song, S.; Burke, K.  
*Quantifying density errors in DFT.*  
**J. Phys. Chem. Lett.** 2018, 9 (22), 6385–6392.  
DOI: [10.1021/acs.jpclett.8b02855](https://doi.org/10.1021/acs.jpclett.8b02855)

Lee, M.; Kim, B.; Sim, M.; Sogal, M.; Kim, Y.; Yu, H.; Burke, K.; Sim, E.  
*Correcting dispersion corrections with density-corrected DFT.*  
**J. Chem. Theory Comput.** 2024, 20 (16), 7155–7167.  
DOI: [10.1021/acs.jctc.4c00689](https://doi.org/10.1021/acs.jctc.4c00689)




 
