# ODT &nbsp;·&nbsp; Organismal Digital Twin

Spatiotemporal foundation model-guided organismal digital twin for
regeneration-regulatory gene discovery in planarians.

---

## Overview

ODT models planarian regeneration at single-cell resolution through five
sequentially connected stages: self-supervised pretraining on unlabelled
scRNA-seq data, spatial-temporal fine-tuning for 3D cell-type annotation,
SVM-based regeneration gene discovery with cross-species BLAST validation,
MLP-driven gene regulatory network inference, and diffusion-model-based
virtual section reconstruction at unobserved timepoints.

---

## Workflow

**1. &nbsp; Geneformer Pretraining**
&nbsp; Masked-language-model pretraining on planarian single-cell transcriptomes
using the Geneformer framework. The resulting foundation model (PFM) learns
contextualised gene embeddings that generalise across downstream tasks.

**2. &nbsp; PFM-T · Regeneration Gene Discovery**
&nbsp; Gene embeddings extracted from the fine-tuned model are classified via
SVM with leave-one-out cross-validation, using experimentally validated
positive and negative gene sets as anchors. Predicted regeneration candidates
are then queried against the NCBI protein database via BLAST, and the
resulting homologues are mapped to taxonomic clades for phylogenetic profiling
and knowledge graph assembly.

**3. &nbsp; PFM-ST · 3D Cell-Type Annotation**
&nbsp; Spatial-temporal enhancement of the pretrained model with a
dual-branch architecture — gene graph attention coupled with dynamic spatial
query — trained on 2D spatial transcriptomics slices. Three-replicate ensemble
prediction with soft-voting is applied to 3D whole-mount planarian data across
seven regeneration timepoints, producing DBSCAN-denoised, publication-quality
3D renderings of nine cell types with eyespot landmark overlay.

**4. &nbsp; GRN · Regulatory Network Construction**
&nbsp; A three-hidden-layer MLP trained on concatenated gene embedding and
perturbation response vectors predicts pairwise regulatory relationships on a
seven-class ordinal scale from strong inhibition to strong activation.
Exhaustive scoring of all ~8.6 million gene pairs identifies high-confidence
directed edges for downstream network analysis and visualisation.

**5. &nbsp; SquiDiff · Virtual Section Reconstruction**
&nbsp; A conditional diffusion model learns the temporal trajectory of
transcriptional programmes in latent space. By encoding real cells and
decoding along a learned temporal direction, SquiDiff generates plausible
gene expression profiles at unobserved intermediate timepoints. The resulting
raw and simulated latent spaces are interrogated through t-SNE and diffusion
pseudotime projections, pairwise cell-type distance dynamics, river plots,
and dynamic network graphs.

---
## Installation

```
bash
# 1. Create and activate the conda environment
conda create -n ODT_main python=3.9
conda activate ODT_main

# 2. Clone the specified branch of Geneformer and install it
git clone --branch pr146_branch https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install -e .

# 3. Return to the project root directory (e.g., /home/wwd/ODT-main)
# Before running the scripts, replace the `CODE_PATH` variable in the first code block
# of each script with the absolute path to your local project directory.
cd /home/wwd/ODT-main   # Replace with your actual project path

# 4. Install the remaining dependencies
pip install -r requirements.txt
```
## Datasets
zenodo [10.5281/zenodo.21212543](https://zenodo.org/records/21212543)

## Results
zenodo [10.5281/zenodo.21618496](https://zenodo.org/records/21618496)

The model weights and Squidiff result files are distributed as multi-volume 7-Zip archives due to file size limitations.

1. Download all files belonging to each archive:

   Model weights:
   - models_split.7z.001
   - models_split.7z.002
   - models_split.7z.003
   ...
   - models_split.7z.011

   Squidiff results:
   - squidiff_results_split.7z.001
   - squidiff_results_split.7z.002
   - squidiff_results_split.7z.003
   ...
   - squidiff_results_split.7z.006

2. Place all volume files of the same archive in the same directory.

3. Extract the archive by opening the first volume (.001) with 7-Zip. The remaining volumes will be detected automatically.

4. The extraction will produce `models.zip` and `squidiff_results.zip` 

5. Extract `models.zip` and `squidiff_results.zip` to obtain the `models/` folder containing the model files and the `squidiff_results/` folder containing the Squidiff results, respectively.
---
## Run

After downloading the `datasets` and `results` folders, run the following commands in the `tutorials` directory.

Or you can visit the webpage https://odt-main.readthedocs.io/en/latest/ to watch the tutorial.

## Citation

```bibtex
@article{wang2025odt,
  title     = {Spatiotemporal foundation model-guided organismal digital twin
               for regeneration-regulatory gene discovery in planarians},
  author    = {Wang, Wendi and ...},
  journal   = {...},
  year      = {2025}
}
```

---

## Contact

Wendi Wang &nbsp;—&nbsp; [wwd23@mails.jlu.edu.cn](mailto:wwd23@mails.jlu.edu.cn)
