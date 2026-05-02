# Density Sensitivity classification using Machine Learning

Density Functional Theory (DFT) is a quantum mechanical method that computes the energy of a system using its electron density instead of its full wavefunction.

For most DFT approximations, errors arise primarily from the functional (functional errors). However, in some systems, errors are dominated by inaccuracies in the electron density (density driven errors). These cases are known as density-sensitive.[[1]](https://dft.uci.edu/projects_DC.php)


To quantify density-driven errors in DFT, the density sensitivity metric $\tilde{S}$ is defined as: 

$\tilde{S} = \left| \tilde{E}[n^{\mathrm{HF}}] - \tilde{E}[n^{\mathrm{LDA}}] \right|$ [[2]](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b02855)

where reactions with $\tilde{s} \ge 2$ kcal/mol are classified as density-sensitive. These cases are considered “abnormal”, and thus Density-Corrected Density Functional Theory (DC-DFT) is expected to improve results. [[1]](https://dft.uci.edu/projects_DC.php)


Computing s̃ requires HF (O(N^4)) and LDA (O(N^3)) densities. We instead use machine learning as a surrogate to classify GMTKN55 benchmark reactions as density-sensitive (s̃ ≥ 2) or density-insensitive (s̃ < 2) without explicitly evaluating s̃.

-This work was done as part of ML research in the Burke Group at UC Irvine.

  
## Project Overview

The pipeline integrates physics-informed molecular encoding with modern ML techniques:

-  **Molecular Parsing** – Uses the *Atomic Simulation Environment (ASE)* to read `.xyz` files and construct `Atoms` objects containing atomic numbers and 3D coordinates. These standardized structures serve as inputs for Molecular descriptor generation.
- **Coulomb Matrix Molecular Descriptor** – Converts each ASE `Atoms` object into a rotation- and permutation-invariant Coulomb matrix molecular descriptor using the `dscribe` library. This descriptor captures interatomic electrostatic interactions in a fixed numerical representation. 
- **Reaction Matrices** – extend molecular descriptors to constructs block-diagonal reaction matrices that account for stoichiometric coefficients of reactant and product molecules.  
- **Spectral Feature Extraction** – Computes and sorts eigenvalues of each reaction matrix to obtain fixed-length, invariant feature vectors.  
- **Learning and Prediction** – Trains **Decision Tree**, **Random Forest** and **XGBoost** models for **binary classification** (density sensitive vs. insensitive).

For a full summary of methods and results, see the [project poster](Density_sensitivity_classification_poster.pdf).


## Project Structure

```
density_sensitivity-classification/
├── Descriptor1/
│   ├── Descriptor1_complete_features.npy           — feature matrix (reaction eigenvalues + metadata)
│   └── Descriptor1_complete_targets.npy            — target labels for reactions (density sensitivity)
│
├── descriptor1_model.ipynb                         — model training and evaluation notebook
├── dimensionality_reduction.ipynb                  — PCA, UMAP, and t-SNE notebook
├── diagonalize_matrices.py                         — computes eigenvalues of reaction matrices
├── generate_cm.py                                  — constructs Coulomb matrices
├── pad_and_metadata.py                             — pads eigenvalue vectors and attaches metadata
├── preprocess.py                                   — preprocessing utility functions
├── main.py                                         — full descriptor generation workflow
├── final_dict_allsets.pkl                          — Coulomb matrices for all GMTKN55 systems
├── Density_sensitivity_classification_poster.pdf   — project poster
├── requirements.txt                                — Python dependencies
└── README.md                                       — documentation

      
```

## Quick Start

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

## Dependencies
ASE, dscribe, NumPy, scikit-learn, XGBoost, Matplotlib.
  
## 📈 Model Performance


The dataset exhibits a moderate class imbalance (~33% density-sensitive vs. ~67% density-insensitive reactions). Models were evaluated using metrics robust to imbalance, including balanced accuracy, recall, and precision.

### Test set performance of each model at its optimal K\* (number of eigenvalues used)

| Model         | K\* | Accuracy | Balanced Accuracy | ROC-AUC | Recall (Minority) | Precision (Minority) |
|---------------|:---:|:--------:|:-----------------:|:-------:|:-----------------:|:--------------------:|
| **XGBoost**       | 22  | **0.821** | **0.812** | **0.883** | 0.784 | **0.710** |
| **Random Forest** | 22  | 0.801 | 0.791 | 0.864 | 0.763 | 0.679 |
| **Decision Tree** | 24  | 0.808 | 0.806 | 0.825 | **0.804** | 0.678 |

---


## Data 

- GMTKN55 database from Goerigk Research Group
- SWARM dataset from Burke Group
  

## 🙏 Acknowledgments

- Burke Group @ UCI
- Goerigk Research Group @ university of Melbourne

## Resources
- <https://hunterheidenreich.com/posts/molecular-descriptor-coulomb-matrix/#the-coulomb-matrix>
- <https://goerigk.chemistry.unimelb.edu.au/research/the-gmtkn55-database>

## Reference

[1] Burke, K.  
*Density-Corrected Density Functional Theory.*  
Burke Research Group, University of California, Irvine.  
https://dft.uci.edu/projects_DC.php

[2] Sim, E.; Song, S.; Burke, K.  
*Quantifying density errors in DFT.*  
**J. Phys. Chem. Lett.** 2018, 9 (22), 6385–6392.  
DOI: [10.1021/acs.jpclett.8b02855](https://doi.org/10.1021/acs.jpclett.8b02855)

[3] Lee, M.; Kim, B.; Sim, M.; Sogal, M.; Kim, Y.; Yu, H.; Burke, K.; Sim, E.  
*Correcting dispersion corrections with density-corrected DFT.*  
**J. Chem. Theory Comput.** 2024, 20 (16), 7155–7167.  
DOI: [10.1021/acs.jctc.4c00689](https://doi.org/10.1021/acs.jctc.4c00689)

<!-- 

[4] Sim, E.; Song, S.; Vuckovic, S.; Kim, Y.; Burke, K.  
*Improving results by improving densities: Density-corrected density functional theory.*  
**J. Am. Chem. Soc.** 2022, 144 (15), 6625–6639.  
DOI: [10.1021/jacs.1c11506](https://doi.org/10.1021/jacs.1c11506)

[4] Goerigk, L.; Hansen, A.; Bauer, C.; Ehrlich, S.; Najibi, A.; Grimme, S.  
*A look at the density functional theory zoo with the advanced GMTKN55 database for general main group thermochemistry, kinetics and noncovalent interactions.*  
**Phys. Chem. Chem. Phys.** 2017, 19, 32184–32215.  
DOI: [10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G)


--> 
