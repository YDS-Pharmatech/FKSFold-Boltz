# Internal hack version for steering integration
# This file contains only the modified parts that cannot be imported from the original diffusionv2.py

from __future__ import annotations

from math import sqrt
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# Import everything else from the original diffusionv2 module
from boltz.model.modules.diffusionv2 import AtomDiffusion as BoltzAtomDiffusion
from boltz.model.modules.utils import default, compute_random_augmentation
from boltz.model.loss.diffusionv2 import weighted_rigid_align

# from boltz_extension.steering.potentials import get_potentials
from boltz_extension.steering.base import (
    should_resample, should_apply_interface_guidance,
    BoltzParticleState
)
from boltz_extension.steering.vis_mixin import BoltzVisualizationMixin
from boltz_extension.steering.trajectory_recorder import BoltzTrajectoryRecorder


class AtomDiffusion(BoltzAtomDiffusion):
    """
    Modified AtomDiffusion with modular steering support.

    This class inherits from the original AtomDiffusion and only overrides
    the sample method to use the new modular steering system.
    """

    def __init__(self, *args, confidence_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_module = confidence_module

        # Trajectory recording components
        self.trajectory_recorder: Optional[BoltzTrajectoryRecorder] = None
        self.visualization_mixin: Optional[BoltzVisualizationMixin] = None

        print(f"[Init] AtomDiffusion: confidence_module {'set' if confidence_module is not None else 'not set'}")

    def set_confidence_module(self, confidence_module):
        """
        Set or update the confidence_module.

        This method allows setting or updating the confidence_module after instance creation,
        which is useful if the confidence module is created after the diffusion module.

        Args:
            confidence_module: The confidence module instance used to compute interface scores.
        """
        self.confidence_module = confidence_module
        print(f"[Update] AtomDiffusion: confidence_module {'set' if confidence_module is not None else 'not set'}")

    def setup_trajectory_recording(
        self,
        output_dir: Optional[Path] = None,
        save_coordinates: bool = True,
        save_confidence_scores: bool = True,
        extra_save_interval: Optional[int] = None,
        enable_visualization: bool = True,
        visualization_output_dir: Optional[Path] = None
    ) -> None:
        """
        Set up trajectory recording functionality.

        Args:
            output_dir: Output directory for trajectory data.
            save_coordinates: Whether to save coordinates.
            save_confidence_scores: Whether to save confidence scores.
            extra_save_interval: Extra coordinate save interval.
            enable_visualization: Whether to enable visualization.
        """
        if output_dir is None:
            from pathlib import Path
            output_dir = Path("./boltz_steering_trajectories")

        # Set up visualization component
        if enable_visualization:
            if visualization_output_dir is None:
                visualization_output_dir = output_dir / "visualizations"
            # Ensure visualization directory exists
            visualization_output_dir.mkdir(parents=True, exist_ok=True)
            self.visualization_mixin = BoltzVisualizationMixin(output_dir=visualization_output_dir)
            print(f"[Trajectory] Visualization component enabled: {visualization_output_dir}")

        # Set up trajectory recorder
        self.trajectory_recorder = BoltzTrajectoryRecorder(
            output_dir=output_dir,
            save_coordinates=save_coordinates,
            save_confidence_scores=save_confidence_scores,
            extra_save_interval=extra_save_interval,
            confidence_module=self.confidence_module,
            confidence_context={}
        )

        # Integrate visualization system
        if self.visualization_mixin:
            self.trajectory_recorder.set_visualization_mixin(self.visualization_mixin)

        print(f"[Trajectory] Trajectory recording system enabled: {output_dir}")
        print(f"[Trajectory] Save coordinates: {save_coordinates}, Save confidence: {save_confidence_scores}")
        print(f"[Trajectory] Extra save interval: {extra_save_interval}")

    def finalize_trajectory_recording(self) -> None:
        """Finalize trajectory recording and save data."""
        if self.trajectory_recorder:
            self.trajectory_recorder.finalize()
            print("[Trajectory] Trajectory recording completed and saved")

    def sample(
        self,
        atom_mask: Tensor,
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1,
        max_parallel_samples: Optional[int] = None,
        steering_args: Optional[Dict[str, Any]] = None,
        interface_steering_args: Optional[Dict[str, Any]] = None,
        **network_condition_kwargs,
    ):
        """
        Modified sample method supporting both physical and interface steering.

        Args:
            steering_args: Physical steering configuration (for backwards compatibility)
            interface_steering_args: Interface steering configuration (dict format)
        """
        # Handle default steering args
        if steering_args is None:
            steering_args = {"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False}

        # Check if interface steering is enabled
        if interface_steering_args is not None and interface_steering_args.get("interface_steering", False):
            # Use interface steering
            return self._sample_with_interface_steering(
                atom_mask=atom_mask,
                num_sampling_steps=num_sampling_steps,
                multiplicity=multiplicity,
                max_parallel_samples=max_parallel_samples,
                interface_args=interface_steering_args,
                **network_condition_kwargs
            )
        else:
            # No steering or physical steering - delegate to original implementation
            # Safely remove confidence_kwargs to avoid passing to parent sample method
            filtered_kwargs = {k: v for k, v in network_condition_kwargs.items()
                               if k != "confidence_kwargs"}
            return super().sample(
                atom_mask=atom_mask,
                num_sampling_steps=num_sampling_steps,
                multiplicity=multiplicity,
                max_parallel_samples=max_parallel_samples,
                steering_args=steering_args,
                **filtered_kwargs
            )

    def _sample_with_interface_steering(
        self,
        atom_mask: Tensor,
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1,
        max_parallel_samples: Optional[int] = None,
        interface_args: Dict[str, Any] = None,
        **network_condition_kwargs,
    ):
        """Sample using interface confidence steering system."""

        confidence_kwargs = network_condition_kwargs.pop("confidence_kwargs", {})
        network_with_confidence_kwargs = {**network_condition_kwargs, "confidence_kwargs": confidence_kwargs}

        # Initialize multiplicity for particle steering
        if interface_args.get("interface_steering"):
            # Check required parameters
            if "num_particles" not in interface_args:
                raise ValueError("interface_steering requires 'num_particles' parameter")
            if "interface_scoring_type" not in interface_args:
                raise ValueError("interface_steering requires 'interface_scoring_type' parameter")
            if "interface_lambda" not in interface_args:
                raise ValueError("interface_steering requires 'interface_lambda' parameter")
            if "interface_resampling_interval" not in interface_args:
                raise ValueError("interface_steering requires 'interface_resampling_interval' parameter")

            num_particles = interface_args["num_particles"]
            multiplicity = multiplicity * num_particles
            confidence_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, num_particles
            )

            # Initialize trajectory recording
            if self.visualization_mixin:
                self.visualization_mixin.initialize_trajectories(num_particles)

            # Print initial configuration info
            print(f"\n[Steering] Init: {num_particles} particles × {multiplicity//num_particles} trajectories")
            print(f"[Steering] Total steps: {num_sampling_steps or 'default'}, using scoring type: {interface_args['interface_scoring_type']}")
            print(f"[Steering] Interface lambda: {interface_args['interface_lambda']}, resampling interval: {interface_args['interface_resampling_interval']}")
            if self.trajectory_recorder:
                print(f"[Steering] Trajectory recording enabled")

        if interface_args.get("interface_guidance"):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )

        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # Get sampling schedule
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # Initialize atom coordinates with noise
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        # Gradually denoise with interface steering
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )
            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if interface_args.get("interface_guidance") and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )

                for sample_ids_chunk in sample_ids_chunks:
                    # Only pass necessary arguments to preconditioned_network_forward
                    clean_kwargs = {k: v for k, v in network_condition_kwargs.items()
                                   if k != 'confidence_kwargs'}

                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **clean_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                # Interface confidence-based resampling
                if should_resample(interface_args, step_idx, num_sampling_steps):
                    # Compute confidence scores using confidence module
                    print(f"\n[Step {step_idx}/{num_sampling_steps}] Computing interface confidence")
                    scoring_type = interface_args["interface_scoring_type"]
                    confidence_scores = self._compute_interface_confidence(
                        atom_coords_denoised,
                        network_with_confidence_kwargs,  # Use dict with confidence_kwargs
                        scoring_type,
                        multiplicity
                    )

                    # Print confidence scores
                    print(f"[Confidence] Scores: {confidence_scores.detach().cpu().numpy().round(4)}")

                    confidence_traj = torch.cat((confidence_traj, confidence_scores.unsqueeze(1)), dim=1)

                    # Record snapshot before resampling (including confidence scores)
                    if self.trajectory_recorder:
                        particles_with_scores = self._create_particle_states(
                            atom_coords=atom_coords,
                            atom_coords_denoised=atom_coords_denoised,
                            num_particles=interface_args["num_particles"],
                            confidence_scores=confidence_scores
                        )

                        # Record state at resampling step
                        self.trajectory_recorder.record_step(
                            step_idx=step_idx,
                            sigma=sigma_t,
                            t_hat=t_hat,
                            steering_t=steering_t,
                            particles=particles_with_scores
                        )

                    # Compute log G values for confidence (higher confidence = better)
                    if step_idx == 0:
                        log_G = confidence_scores  # Maximize confidence
                    else:
                        log_G = confidence_traj[:, -1] - confidence_traj[:, -2]  # Improvement in confidence
                        print(f"[Confidence] Score change: {log_G.detach().cpu().numpy().round(4)}")

                    # Compute ll difference for guidance
                    if interface_args.get("interface_guidance") and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(confidence_scores)

                    # Compute resampling weights based on confidence
                    interface_lambda = interface_args["interface_lambda"]
                    num_particles = interface_args["num_particles"]
                    resample_weights = F.softmax(
                        (ll_difference + interface_lambda * log_G).reshape(
                            -1, num_particles
                        ),
                        dim=1,
                    )

                    # Print resampling weights
                    print(f"[Resampling] Resampling weights: {resample_weights.detach().cpu().numpy().round(3)}")

                    # Print full debug info
                    self._debug_print_steering_info(
                        step_idx=step_idx,
                        num_sampling_steps=num_sampling_steps,
                        confidence_scores=confidence_scores,
                        log_G=log_G,
                        resample_weights=resample_weights
                    )

                # # Interface confidence guidance
                # if should_apply_interface_guidance(interface_args, step_idx, num_sampling_steps):
                #     guidance_update = torch.zeros_like(atom_coords_denoised)
                #     interface_gd_steps = interface_args.get("interface_gd_steps")
                #     if interface_gd_steps is None:
                #         raise ValueError("interface_guidance requires 'interface_gd_steps' parameter")
                #     print(f"\n[Guidance] Starting gradient guidance, {interface_gd_steps} steps")
                #
                #     for guidance_step in range(interface_gd_steps):
                #         # Compute confidence gradient
                #         scoring_type = interface_args["interface_scoring_type"]
                #         confidence_gradient = self._compute_confidence_gradient(
                #             atom_coords_denoised + guidance_update,
                #             network_with_confidence_kwargs,  # Use dict with confidence_kwargs
                #             scoring_type,
                #             multiplicity
                #         )
                #         interface_guidance_strength = interface_args.get("interface_guidance_strength")
                #         if interface_guidance_strength is None:
                #             raise ValueError("interface_guidance requires 'interface_guidance_strength' parameter")
                #         guidance_update += interface_guidance_strength * confidence_gradient
                #
                #         # Print gradient info (first and last step)
                #         if guidance_step == 0 or guidance_step == interface_gd_steps - 1:
                #             grad_norm = torch.norm(confidence_gradient, dim=(-1,-2)).mean().item()
                #             print(f"[Guidance] Step {guidance_step+1}/{interface_gd_steps}, grad norm: {grad_norm:.6f}")
                #
                #     atom_coords_denoised += guidance_update
                #
                #     # Print final update magnitude
                #     guidance_magnitude = torch.norm(guidance_update, dim=(-1,-2)).mean().item()
                #     print(f"[Guidance] Final update magnitude: {guidance_magnitude:.6f}")
                #     scaled_guidance_update = (
                #         guidance_update
                #         * self.step_scale
                #         * (sigma_t - t_hat)
                #         / t_hat
                #     )

                # Apply resampling if needed
                if should_resample(interface_args, step_idx, num_sampling_steps):
                    print(f"\n[Resampling] Performing resampling (step_idx={step_idx})")

                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    # Print resampling result
                    flat_indices = resample_indices.detach().cpu().numpy().flatten()
                    source_indices = np.arange(len(flat_indices))
                    print(f"[Resampling] Source idx -> Target idx: {list(zip(source_indices, flat_indices))}")

                    # Count how many times each particle is selected
                    counts = np.bincount(flat_indices)
                    selected_counts = [(i, count) for i, count in enumerate(counts) if count > 0]
                    print(f"[Resampling] Path selection stats: {selected_counts}")

                    # Record resampling snapshot (before actual resampling)
                    if self.trajectory_recorder and 'particles_with_scores' in locals():
                        # Convert numpy types to Python native types for JSON serialization
                        resampling_mapping = {int(i): int(flat_indices[i]) for i in range(len(flat_indices))}
                        self.trajectory_recorder.record_resampling_snapshot(
                            step_idx=step_idx,
                            sigma=sigma_t,
                            t_hat=t_hat,
                            steering_t=steering_t,
                            particles=particles_with_scores,
                            resampling_mapping=resampling_mapping
                        )

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    confidence_traj = confidence_traj[resample_indices]
                    if interface_args.get("interface_guidance"):
                        scaled_guidance_update = scaled_guidance_update[resample_indices]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        # Finalize trajectory recording
        if self.trajectory_recorder:
            self.finalize_trajectory_recording()

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)

    def _create_particle_states(
        self,
        atom_coords: Tensor,
        atom_coords_denoised: Optional[Tensor],
        num_particles: int,
        confidence_scores: Optional[Tensor] = None,
        token_repr: Optional[Tensor] = None
    ) -> List[BoltzParticleState]:
        """Create a list of BoltzParticleState objects for trajectory recording."""
        particles = []

        for i in range(num_particles):
            # Extract coordinates for a single particle
            particle_coords = atom_coords[i:i+1] if atom_coords is not None else None
            particle_coords_denoised = atom_coords_denoised[i:i+1] if atom_coords_denoised is not None else None
            particle_token_repr = token_repr[i:i+1] if token_repr is not None else None

            # Extract confidence score for a single particle
            particle_interface_score = None
            if confidence_scores is not None and i < len(confidence_scores):
                particle_interface_score = float(confidence_scores[i].item())

            particle = BoltzParticleState(
                atom_coords=particle_coords,
                atom_coords_denoised=particle_coords_denoised,
                token_repr=particle_token_repr,
                interface_score=particle_interface_score
            )
            particles.append(particle)

        return particles

    def _compute_interface_confidence(
        self,
        atom_coords_denoised: Tensor,
        network_condition_kwargs: Dict[str, Any],
        scoring_type: str,
        multiplicity: int
    ) -> Tensor:
        """Compute interface confidence scores using the confidence module."""
        feats = network_condition_kwargs["feats"]
        if self.confidence_module is None:
            raise ValueError("Confidence module is not set!")

        confidence_kwargs = network_condition_kwargs.get("confidence_kwargs")
        confidence_output = self.confidence_module(
            s_inputs=confidence_kwargs.get("s_inputs"),
            s=confidence_kwargs.get("s"),
            z=confidence_kwargs.get("z"),
            x_pred=atom_coords_denoised,
            feats=feats,
            pred_distogram_logits=confidence_kwargs.get("pred_distogram_logits"),
            multiplicity=multiplicity
        )

        print(f"[Debug] confidence_output type: {type(confidence_output).__name__}")
        print(f"[Debug] confidence_output keys: {list(confidence_output.keys()) if isinstance(confidence_output, dict) else 'not a dict'}")

        if scoring_type in confidence_output:
            score = confidence_output[scoring_type]
            print(f"[Score] Using {scoring_type} score: {score.mean().item():.4f}")
            return score
        else:
            raise ValueError(f"Required interface score {scoring_type} not found in confidence_output")

    def _debug_print_steering_info(self, step_idx, num_sampling_steps, confidence_scores, log_G=None, resample_weights=None, resample_indices=None):
        """Print detailed interface steering debug information."""
        print("\n" + "="*50)
        print(f"[Debug Info] Step {step_idx}/{num_sampling_steps}")
        print("-"*50)

        if confidence_scores is not None:
            print("Confidence scores:")
            for i, score in enumerate(confidence_scores.detach().cpu().numpy()):
                print(f"  Particle {i}: {score:.4f}")

        if log_G is not None:
            print("Objective values (log_G):")
            for i, lg in enumerate(log_G.detach().cpu().numpy()):
                print(f"  Particle {i}: {lg:.4f}")

        if resample_weights is not None:
            print("Resampling weights:")
            weights = resample_weights.detach().cpu().numpy()
            if len(weights.shape) > 1:
                for i, w_group in enumerate(weights):
                    print(f"  Group {i}: {w_group.round(3)}")
            else:
                print(f"  {weights.round(3)}")

        if resample_indices is not None:
            indices = resample_indices.detach().cpu().numpy().flatten()
            source_indices = np.arange(len(indices))
            mapping = list(zip(source_indices, indices))
            print("Resampling mapping (source -> target):")
            for src, dst in mapping:
                print(f"  {src} -> {dst}")

            # Count how many times each index was selected
            counts = np.bincount(indices)
            print("Selection stats:")
            for i, count in enumerate(counts):
                if count > 0:
                    print(f"  Index {i}: selected {count} times")

        print("="*50 + "\n")

    def _compute_confidence_gradient(
        self,
        atom_coords: Tensor,
        network_condition_kwargs: Dict[str, Any],
        scoring_type: str,
        multiplicity: int
    ) -> Tensor:
        """Compute gradient of confidence score with respect to coordinates."""
        # Enable gradient computation
        atom_coords.requires_grad_(True)

        try:
            # Compute confidence score
            confidence_scores = self._compute_interface_confidence(
                atom_coords, network_condition_kwargs, scoring_type, multiplicity
            )

            # Compute total confidence (sum over all particles)
            total_confidence = confidence_scores.sum()

            # Compute gradient
            gradient = torch.autograd.grad(
                total_confidence,
                atom_coords,
                create_graph=False,
                retain_graph=False
            )[0]

            return gradient

        except Exception:
            # Fallback to zero gradient if computation fails
            return torch.zeros_like(atom_coords)
        finally:
            # Reset gradient requirement
            atom_coords.requires_grad_(False)
