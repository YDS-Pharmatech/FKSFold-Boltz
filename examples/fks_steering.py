import sys
from pathlib import Path
import torch

# Import the main module
from boltz.main import predict_core


input_file = "./multimer.fasta"
output_dir = "../outputs/interface_steering"

def test_ptm_steering():
    """Run diffusion with interface confidence steering"""
        
    # Run inference with interface confidence steering
    predict_core(
        data=input_file,
        out_dir=output_dir,
        # Interface steering parameters
        use_potentials=False,  # No physical steering
        use_interface_steering=True,  # Enable interface steering
        # interface_scoring_type="mean_iptm",  # Focus on interface PTM
        interface_scoring_type="protein_mean_iptm",  # Focus on interface PTM
        interface_lambda=2.0,
        interface_resampling_interval=5,
        interface_gd_steps=10,
        num_particles=3,
        sampling_steps=200,
        step_scale=1.0,
        subsample_msa=False,  # CLI default vs predict_core default True
        use_msa_server=True,
        override=True,
    )


def main():
    test_ptm_steering()


if __name__ == "__main__":
    main() 