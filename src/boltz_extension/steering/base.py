from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from torch import Tensor
from pathlib import Path

@dataclass 
class BoltzParticleState:
    """State of a single particle in Boltz iPTM-based steering"""
    atom_coords: Tensor  # Current atom coordinates [num_atoms, 3]
    atom_coords_denoised: Optional[Tensor] = None  # Denoised coordinates from network
    token_repr: Optional[Tensor] = None  # Token representations if needed
    
    # iPTM-based scoring (no physical energy for Boltz interface steering)
    interface_score: Optional[float] = None  # Current interface confidence score (iPTM, mean_iPTM, etc.)
    historical_score: Optional[float] = None  # Historical score for potential DIFF-style steering
    score_trajectory: List[float] = field(default_factory=list)  # Complete score history


@dataclass
class BoltzTrajectoryPoint:
    """Record state at a specific time point during Boltz diffusion process"""
    step: int
    sigma: float
    t_hat: float  # Actual timestep used in boltz
    steering_t: float  # Steering time parameter (1.0 - step/total_steps)
    
    # iPTM-based scoring
    interface_score: Optional[float] = None  # Interface confidence score at this step
    confidence_scores: Optional[Dict[str, float]] = None  # Detailed confidence scores (iPTM, mean_iPTM, etc.)
    
    # Enhanced fields for coordinate tracking  
    atom_coords: Optional[Tensor] = None  # Current coordinates [num_atoms, 3]
    atom_coords_denoised: Optional[Tensor] = None  # Denoised coordinates
    
    # Metadata flags
    is_resampling_point: bool = False
    is_extra_saved_point: bool = False  # Whether this is an extra saved point
    
    def has_coordinates(self) -> bool:
        """Check if coordinates are available"""
        return self.atom_coords is not None


@dataclass
class BoltzParticleTrajectory:
    """Record complete trajectory of a single particle in Boltz diffusion"""
    particle_id: int
    points: List[BoltzTrajectoryPoint] = field(default_factory=list)
    resampled_from: List[int] = field(default_factory=list)  # Record resampling history
    
    def get_resampling_points(self) -> List[BoltzTrajectoryPoint]:
        """Get all resampling points in this trajectory"""
        return [point for point in self.points if point.is_resampling_point]
    
    def get_extra_saved_points(self) -> List[BoltzTrajectoryPoint]:
        """Get all extra saved points in this trajectory"""
        return [point for point in self.points if point.is_extra_saved_point]
    
    def get_points_with_coordinates(self) -> List[BoltzTrajectoryPoint]:
        """Get all points that have coordinate data"""
        return [point for point in self.points if point.has_coordinates()]


@dataclass
class BoltzInterfaceSteeringConfig:
    """Configuration for pure iPTM-based interface steering"""
    
    # Core iPTM steering parameters
    interface_steering: bool = True
    interface_scoring_type: str = "mean_iptm"
    interface_lambda: float = 2.0  # iPTM steering weight
    interface_resampling_interval: int = 5  # Resampling every N steps
    interface_gd_steps: int = 10  # Number of gradient descent steps
    num_particles: int = 3  # Number of particles for steering
    
    # Interface guidance parameters
    interface_guidance: bool = False  # Enable confidence gradient guidance
    interface_guidance_strength: float = 0.1  # Strength of gradient guidance
    
    # Trajectory recording options
    enable_trajectory_recording: bool = False  # Enable trajectory recording
    trajectory_output_dir: Optional[Path] = None  # Directory to save trajectory data
    save_coordinates: bool = True  # Whether to save coordinates
    save_confidence_scores: bool = True  # Whether to save confidence scores
    
    # Visualization options
    enable_visualization: bool = False  # Enable visualization
    visualization_output_dir: Optional[Path] = None  # Directory for visualization outputs
    

# Standalone helper functions for interface steering logic
def should_resample(interface_args: Dict[str, Any], step_idx: int, num_sampling_steps: Optional[int] = None) -> bool:
    """Check if interface steering should run at this step"""
    if not interface_args.get("interface_steering"):
        return False
    
    interface_resampling_interval = interface_args.get("interface_resampling_interval")
    if interface_resampling_interval is None:
        raise ValueError("interface_steering requires 'interface_resampling_interval' parameter")
    
    # Normal interval-based resampling
    regular_resampling = (step_idx % interface_resampling_interval == 0)
    
    # Final step resampling for selecting final particle
    final_step_resampling = (num_sampling_steps is not None and 
                            step_idx == num_sampling_steps - 1)
    
    should_run = regular_resampling or final_step_resampling
    
    # Print log for resampling
    if should_run:
        reason = "Regular interval resampling" if regular_resampling else "Final step resampling"
        print(f"[Steering] Step {step_idx}: Performing interface resampling ({reason})")
    
    return should_run


def should_apply_interface_guidance(interface_args: Dict[str, Any], step_idx: int, num_sampling_steps: int) -> bool:
    """Check if interface guidance should be applied"""
    interface_guidance = interface_args.get("interface_guidance")
    should_apply = (interface_guidance and step_idx < num_sampling_steps - 1)
    
    # Print log for interface guidance
    if should_apply:
        print(f"[Steering] Step {step_idx}: Applying interface gradient guidance")
    
    return should_apply
