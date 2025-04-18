# A context-aware genomic language model for gene spatial expression imputation, pattern detection, and function discovery

<p align="justify">
We develop <strong>SIGEL</strong>, a pioneer cost-effective self-supervised language model that generates gene manifolds from spatial transcriptomics data through exploiting spatial genomic “context” identified through spatial expression relationships among genes. <strong>S</strong>IGEL-<strong>G</strong>enerated gene <strong>R</strong>epresentations (<strong>SGEs</strong>) feature in context-awareness, rich semantics, and robustness to cross-sample technical artifacts. Extensive analyses of real data have demonstrated the biological relevance of the genomic contexts identified by SIGEL, confirming the functional and relational semantics of SGRs. Moreover, SGRs can be applied to a variety of key downstream analytical objectives in biomedical research.
</p>

## Outline
<p align="justify">
ST data reveal spatial genomic contexts comprising genes cofunctional in gene pathways and networks since they tend to exhibit similar expression patterns across tissue space. By leveraging these genomic contexts, distributed gene representations can be learned. These representations not only encapsulate gene spatial functional and relational semantics but also instrumental in facilitating various downstream task-specific objectives.
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/7add660b-cd47-48ca-89be-2b4ea561023b" width="1000">
</p>

## Framework of SIGEL
<p align="justify">
SIGEL is an innovative self-supervised learning method that represents genes as distributed manifolds, capturing both spatial gene expression and gene co-expression information through masked autoencoder and boasting clustering. The framework comprises three key modules:

<i>Module I</i> employs an adapted masked autoencoder to learn the representations of gene images, which enhances gene embeddings’ local-context perceptibility.

<i>Module II</i> involves modeling gene embeddings using a Student’s t mixture model, with parameters estimated via a MAP-EM algorithm. This module aims to maximize the likelihood of the entire dataset.

<i>Module III</i> refines gene embeddings through a self-paced pretext task designed to identify genomic contexts via iterative pseudo-contrastive learning. Together, Modules II and III complete a single training epoch, during which the discriminability of gene embeddings is significantly enhanced.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d319e402-8e2a-4eb7-a572-c9cb8cb950a9" width="1000">
</p>


## Dependencies
- Python >=3.9.15
- torch>=1.13.0
- rpy2>=3.5.13
- scikit-learn>=1.2.0
- scanpy>=1.9.6
- scipy>=1.11.4
- pandas>=1.5.2
- numpy>=1.21.6
- sympy>=1.11.1
- anndata>=0.10.3
- SpaGCN>=1.2.7
- tqdm>=4.64.1

## Installation
You can download the package from GitHub and install it locally:
```bash
git clone https://github.com/WLatSunLab/SIGEL.git
cd SIGEL/
python3 setup.py install
```

## Sample data
Sample data including 10x-hDLPFC-151676, 10x-mEmb, seq-mEmb can be found [here](https://drive.google.com/drive/folders/1C3Gk-HVYp2dQh4id8H68M9p8IWEOIut_?usp=drive_link) and make sure these data are organized in the following structure:
```
 . <SIGEL>
        ├── ...
        ├── <data>
        │   ├── 151676_10xvisium.h5ad
        │   ├── DLPFC_matrix_151676.dat
        │   └── <mEmb>
        │       ├── 10x_mEmb_matrix.dat
        │       ├── sqf_mEmb_adata.h5ad
        │       └── qf_mEmb_matrix.dat
        ├── <model_pretrained>
        │   │
        └── ...

```

## Getting Started
SIGEL offers a variety of functionalities, including but not limited to:
- Imputing missing genes in FISH-based ST to enhance transcriptomic coverage ([tutorial](https://zipging.github.io/SIGEL.github.io/#3-imputing-missing-genes-in-fish-based-st-to-enhance-transcriptomic-coverage))
- Detecting genes with spatial expression patterns ([tutorial](https://zipging.github.io/SIGEL.github.io/#4-spatial-variability-genes-detection))

  -Co-expression genes
  -Spatial variability genes detection
  -Designated expression patterns across specific tissue regions

- Identifying disease-associated genes and gene-gene interactions across multiple samples. ([tutorial](https://zipging.github.io/SIGEL.github.io/))
- Improving spatial clustering ([tutorial](https://zipging.github.io/SIGEL.github.io/))

<p align="justify">
Before starting the SIGEL tutorial, we need to make some necessary preparations, including installing SIGEL and its required Python dependencies, and downloading the datasets needed for this tutorial. The specific preparation steps can be found in the <a href="https://zipging.github.io/SIGEL.github.io/#1-preparation">SIGEL Preparation Guide</a>. Additionally, considering that SIGEL will process a substantial amount of image data, we strongly recommend using a GPU (e.g., an RTX 3090) to pretrain SIGEL. Doing so can significantly accelerate the processing speed of SIGEL during both the training and application phases.
</p>


## Getting help
If you have any questions or requirements, feel free to reach out to the repository maintainer at zipging@gmail.com.


## Citation
Under review.
