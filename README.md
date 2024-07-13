# A context-aware genomic language model for gene spatial expression imputation, pattern detection, and function discovery


We develop **SIGEL**, a pioneer cost-effective self-supervised language model that generates gene manifolds from spatial transcriptomics data through exploiting spatial genomic “context” identified through spatial expression relationships among genes. **S**IGEL-**G**enerated gene **R**epresentations (**SGR**) feature in context-awareness, rich semantics, and robustness to cross-sample technical artifacts. Extensive analyses of real data have demonstrated the biological relevance of the genomic contexts identified by SIGEL, confirming the functional and relational semantics of SGRs. Moreover, SGRs can be applied to a variety of key downstream analytical objectives in biomedical research.


## Outline
ST data reveal spatial genomic contexts comprising genes cofunctional in gene pathways and networks since they tend to exhibit similar expression patterns across tissue space. By leveraging these genomic contexts, distributed gene representations can be learned. These representations not only encapsulate gene spatial functional and relational semantics but also instrumental in facilitating various downstream task-specific objectives.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b284de6b-ed82-4f00-9e05-a66de8469145" width="900">
</p>

## Framework of SIGEL
SIGEL is an innovative self-supervised learning method that represents genes as distributed manifolds, capturing both spatial gene expression and gene co-expression information through masked autoencoder and boasting clustering. The framework comprises three key modules:

*Module I* employs an adapted masked autoencoder to learn the representations of gene images, which enhances gene embeddings’ local-context perceptibility.

*Module II* involves modeling gene embeddings using a Student’s t mixture model, with parameters estimated via a MAP-EM algorithm. This module aims to maximize the likelihood of the entire dataset.

*Module III* refines gene embeddings through a self-paced pretext task designed to identify genomic contexts via iterative pseudo-contrastive learning. Together, Modules II and III complete a single training epoch, during which the discriminability of gene embeddings is significantly enhanced.







## Dependencies
```
[Python 3.9.15]
[torch 1.13.0]
[rpy2 3.5.13]
[sklearn 1.2.0]
[scanpy 1.9.3]
[scipy 1.9.3]
[pandas 1.5.2]
[numpy 1.21.6]
[sympy 1.11.1]
[SpaGCN 1.2.7]
[anndata 0.10.3]
```

## Applicable tasks
```
* Enhancement of the transcriptomic coverage.
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains.
...
```

## Installation
You can download the package from GitHub and install it locally:
```bash
git clone https://github.com/WLatSunLab/SIGEL.git
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
The [tutorial](https://zipging.github.io/SIGEL.github.io/) included in the repository provides guidance on how to effectively utilize SIGEL and it is continually being refined to better present the paper.

# Others
If you have any questions or requirements, feel free to reach out to the repository maintainer, Wenlin Li, at zipging@gmail.com.
