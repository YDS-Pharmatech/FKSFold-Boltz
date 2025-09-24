from boltz.main import predict_core


input_file = "./multimer.fasta"
output_dir = "../outputs/interface_steering"

def test_baseline_diffusion():
    """Run baseline diffusion without any steering"""
    
    predict_core(
        data=input_file,
        out_dir=output_dir,
        use_potentials=False,
        sampling_steps=200,
        step_scale=1.0,
        subsample_msa=False,
        use_msa_server=True,
        override=True,
    )


def main():
    test_baseline_diffusion()


if __name__ == "__main__":
    main()