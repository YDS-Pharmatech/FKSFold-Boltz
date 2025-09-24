# FKSFold-Boltz

FKSFold-Boltz applies Feynman-Kac (FK) steering to guide the diffusion process in AlphaFold3-type models for molecular glue induced ternary structure prediction. This repository contains the implementation of our early approach that we explored before developing [YDS-GlueFold](https://www.biorxiv.org/content/10.1101/2024.12.23.630090v3), our more comprehensive and successful model for predicting molecular glue ternary complexes.


## Installation

Install from source:

```bash
git clone https://github.com/YDS-Pharmatech/FKSFold-Boltz.git
cd FKSFold-Boltz; pip install -e .[cuda]
```

> Note: we recommend installing fksfold.boltz in a fresh python environment

## Usage

Our usage is mostly compatible with the [Boltz](https://github.com/jwohlwend/boltz) repo. We added FK steering parameters to guide the diffusion process for better ternary complex prediction. You can reference to Boltz's [README](https://github.com/jwohlwend/boltz/blob/main/README.md) for more details.

### Pythonic inference

```shell
python examples/fks_steering.py
```

### Command line inference

For more details, please refer to the [Boltz](https://github.com/jwohlwend/boltz/blob/main/README.md) repo and using `boltz-fks predict --help` to see all the available options.

You can fold a FASTA file containing all the sequences (including proteins and ligands as SMILES strings) in a complex of interest by calling:

```shell
boltz-fks predict input.fasta --use_msa_server
```

### Steering-Enhanced Inference

For enhanced performance on molecular glue ternary complexes, you can use FK steering:

```shell
boltz-fks predict input.fasta \
    --use_interface_steering \
    --interface_scoring_type protein_mean_iptm \
    --interface_lambda 2.0 \
    --interface_resampling_interval 5 \
    --interface_gd_steps 10 \
    --num_particles 3 \
    --use_msa_server
```


## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.

## Citation

The FKSFold-Boltz repo is highly based on the [Boltz](https://github.com/jwohlwend/boltz) repo. If you found this repo useful, please cite the following:

```bibtex
@article{FKSFold-Technical-Report,
	title        = {FKSFold: Improving AlphaFold3-Type Predictions of Molecular Glue-Induced Ternary Complexes with Feynman-Kac-Steered Diffusion},
	author       = {Shen, Jian and Zhou, Shengmin and Che, Xing},
	year         = 2025,
	journal      = {bioRxiv},
	publisher    = {Cold Spring Harbor Laboratory},
	doi          = {10.1101/2025.05.03.651455},
	url          = {https://www.biorxiv.org/content/10.1101/2025.05.03.651455v1},
	elocation-id = {2025.05.03.651455},
	eprint       = {https://www.biorxiv.org/content/10.1101/2025.05.03.651455v1.full.pdf}
}

@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

Additionally, if you use the automatic MMseqs2 MSA generation described above, please also cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
