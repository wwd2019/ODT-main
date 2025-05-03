# ODT
Organismal Digital Twin (ODT) is a digital twin framework designed to model and simulate the regeneration process of planarians (flatworms) at the cellular and molecular levels. Planarians are known for their remarkable regenerative abilities, and ODT seeks to provide a computational platform to understand, analyze, and predict these biological processes.

# Model Description
The Planarian Foundation Model (PFM) is a pretrained transformer model developed to understand gene expression dynamics in planarian regeneration. Due to the limited size and quality of available planarian data, PFM leverages knowledge transfer from the human-trained Geneformer model, which was pretrained on 3.6 million single-cell profiles.

PFM adapts Geneformer by replacing human gene tokens with planarian-specific tokens and retraining on over 500,000 planarian cells in a self-supervised manner. This enables the model to learn gene-gene relationships and molecular patterns specific to planarians, while benefiting from the structure and knowledge learned from human data.

To make the model regeneration-aware, we fine-tuned it using over 130,000 annotated planarian cells covering all major regeneration stages. The model demonstrated strong performance on regeneration phase classification tasks, even with limited labels.

In addition to temporal features, PFM also integrates spatial information by embedding each cell's position in tissue, allowing it to learn spatiotemporal characteristics of planarian regeneration.
# Application

🧬 Discovery of Novel Regeneration-Related Genes
Powered by high-dimensional representations extracted by the Planarian Foundation Model (PFM), ODT enables the identification of candidate genes showing key expression changes across different regeneration stages under unsupervised or weakly supervised settings. This facilitates new insights into the mechanisms of planarian regeneration.

🧠 Gene Regulatory Network Construction
ODT models co-expression relationships between genes to infer potential regulatory networks. By leveraging attention mechanisms and graph neural networks, ODT can generate biologically meaningful and directed regulatory pathways, aiding the discovery of core regulatory factors and modules.

🧭 Cross-Species Transfer Learning
ODT incorporates large-scale pretraining on human data and transfers the learned knowledge to planarians through structural adaptation and token remapping. This greatly improves model performance under limited data conditions specific to non-model organisms.

🧱 3D Data Reconstruction and Visualization
ODT supports the reconstruction of three-dimensional structures from slice-level single-cell spatial transcriptomics. An interactive interface allows users to visualize the spatial distribution of specific genes in 3D space, providing an intuitive understanding of spatial expression patterns.

🧩 Flexible Model Extension
The modular framework of ODT enables flexible fine-tuning for various downstream tasks such as cell type classification or regeneration stage prediction. This makes ODT adaptable for a wide range of applications in regeneration biology and stem cell research.

# Installation
The pretrained models and datasets are hosted on a cloud storage platform. You can download them using the following links:

    Pretrained Models: https://pan.baidu.com/s/1Eo7iExqAkXwyy40IRNyAtQ?pwd=2e26

    Datasets: https://pan.baidu.com/s/1ahePN18BViMPZE25Nt0wrw?pwd=a6jf

    Results: https://pan.baidu.com/s/1W17q7hyiu49k6-CD_i9t6Q?pwd=gqfd

Once downloaded, place the files in the appropriate directories:

    Place the pretrained models in the ./models/ directory.

    Place the datasets in the ./dataSets/ directory.

Note: Ensure that the data files are unzipped and placed correctly for the code to access them.
```bash
git clone https://github.com/yourusername/ODT-main.git
cd ODT-main
```
# quick start
You can quickly start training the Planarian Foundation Model (PFM) using the main.py script, which allows for easy initiation of the training process with the default settings.

Alternatively, you can refer to the example directory for more details and advanced use cases.
