from pathlib import Path
from boltz.main import predict_core


input_file = "./multimer.fasta"
output_dir = "../outputs/interface_steering_vis"

def test_ptm_steering_with_trajectory_recording():
    """Run diffusion with interface confidence steering and trajectory recording"""
    
    predict_core(
        data=input_file,
        out_dir=output_dir,
        use_potentials=False,
        use_interface_steering=True,
        interface_scoring_type="protein_mean_iptm",
        interface_lambda=2.0,
        interface_resampling_interval=5,
        interface_gd_steps=10,
        num_particles=3,
        sampling_steps=50,
        step_scale=1.0,
        enable_trajectory_recording=True,
        trajectory_output_dir=Path(output_dir) / "trajectories",
        enable_visualization=True,
        visualization_output_dir=Path(output_dir) / "trajectories" / "visualizations",
        subsample_msa=False,
        use_msa_server=True,
        override=True,
    )


def create_visualization_from_saved_data(output_dir: str):
    from boltz_extension.steering import BoltzVisualizationMixin
    
    trajectory_dir = Path(output_dir) / "trajectories"
    visualizations_dir = trajectory_dir / "visualizations"
    
    score_csv = visualizations_dir / "interface_score_trajectories.csv"
    resampling_csv = visualizations_dir / "resampling_events.csv"
    
    analysis_output = Path(output_dir) / "analysis"
    analysis_output.mkdir(parents=True, exist_ok=True)
    viz = BoltzVisualizationMixin(output_dir=analysis_output)
    
    viz.load_from_csv(
        score_csv_path=str(score_csv),
        resampling_csv_path=str(resampling_csv) if resampling_csv.exists() else None,
        experiment_name="boltz_interface_steering"
    )
    
    fig1 = viz.plot_interface_score_trajectories(
        title="Boltz Interface iPTM Steering - Score Trajectories"
    )
    trajectory_plot = analysis_output / "interface_score_trajectories.png"
    fig1.savefig(trajectory_plot, dpi=300, bbox_inches='tight')
    
    if resampling_csv.exists():
        fig2 = viz.plot_resampling_flow(
            title="Boltz Interface Steering - Particle Resampling Flow"
        )
        resampling_plot = analysis_output / "resampling_flow.png"
        fig2.savefig(resampling_plot, dpi=300, bbox_inches='tight')


def compare_with_baseline_experiment():
    from boltz_extension.steering import BoltzVisualizationMixin
    
    comparison_output = Path("./outputs/comparison_analysis")
    comparison_output.mkdir(parents=True, exist_ok=True)
    viz = BoltzVisualizationMixin(output_dir=comparison_output)
    
    trajectory_path = "./outputs/03b_interface_steering_with_trajectory/trajectories/visualizations"
    if Path(trajectory_path).exists():
        viz.load_from_csv(
            score_csv_path=f"{trajectory_path}/interface_score_trajectories.csv",
            resampling_csv_path=f"{trajectory_path}/resampling_events.csv",
            experiment_name="with_trajectory_recording"
        )


def main():
    test_ptm_steering_with_trajectory_recording()
    create_visualization_from_saved_data(output_dir)
    compare_with_baseline_experiment()


if __name__ == "__main__":
    main()