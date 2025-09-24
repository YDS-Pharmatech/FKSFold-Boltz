from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import torch
from torch import Tensor
import logging
import numpy as np

from boltz_extension.steering.base import BoltzTrajectoryPoint, BoltzParticleTrajectory, BoltzParticleState


class BoltzTrajectoryRecorder:
    """Boltz diffusion trajectory recorder with coordinate and interface confidence scoring capabilities"""
    
    def __init__(self,
                 output_dir: Path,
                 save_coordinates: bool = True,
                 save_confidence_scores: bool = True,
                 extra_save_interval: Optional[int] = None,
                 confidence_module = None,
                 confidence_context: Optional[Dict] = None):
        """
        Initialize Boltz trajectory recorder
        
        Args:
            output_dir: Directory to save trajectory data
            save_coordinates: Whether to save coordinates
            save_confidence_scores: Whether to save interface confidence scores
            extra_save_interval: Interval for extra coordinate saves (e.g., every 3 steps)
            confidence_module: Model component for confidence calculation (iPTM, mean_iPTM, etc.)
            confidence_context: Context data needed for confidence calculation
        """
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_coordinates = save_coordinates
        self.save_confidence_scores = save_confidence_scores
        self.extra_save_interval = extra_save_interval
        
        # Interface confidence computation setup (iPTM-based)
        self.confidence_module = confidence_module
        self.confidence_context = confidence_context or {}
        
        # Data storage
        self.trajectories: Dict[int, BoltzParticleTrajectory] = {}
        self.resampling_events: List[Dict] = []
        
        # Integration with visualization
        self.vis_mixin = None
        
        logging.info(f"BoltzTrajectoryRecorder initialized: {self.output_dir}")
        logging.info(f"Save coordinates: {save_coordinates}, Save confidence scores: {save_confidence_scores}")
        logging.info(f"Extra save interval: {extra_save_interval}")
    
    def set_visualization_mixin(self, vis_mixin) -> None:
        """Integrate with visualization system"""
        self.vis_mixin = vis_mixin
        logging.info("Visualization mixin integrated with BoltzTrajectoryRecorder")
        
    def record_step(self, 
                   step_idx: int, 
                   sigma: float,
                   t_hat: float,
                   steering_t: float,
                   particles: List[BoltzParticleState],
                   confidence_scores: Optional[Dict[str, float]] = None,
                   is_extra_save: bool = False,
                   is_guidance_applied: bool = False) -> None:
        """Record current step state with Boltz-specific parameters"""
        
        for i, particle in enumerate(particles):
            # Decide whether to save coordinates
            save_coords = self.save_coordinates and (
                is_extra_save or 
                (self.extra_save_interval and step_idx % self.extra_save_interval == 0)
            )
            
            # Use pre-computed confidence scores if available, otherwise try to compute
            interface_confidence_scores = confidence_scores
            if self.save_confidence_scores and not interface_confidence_scores and self.confidence_module:
                try:
                    interface_confidence_scores = self._compute_interface_confidence_scores(particle.atom_coords)
                except Exception as e:
                    logging.warning(f"Failed to compute interface confidence scores at step {step_idx}: {e}")
            
            # Create trajectory point - handle potential CUDA errors
            try:
                # Clone coordinates safely
                atom_coords_clone = None
                atom_coords_denoised_clone = None
                
                if save_coords:
                    if particle.atom_coords is not None:
                        atom_coords_clone = self._safe_clone_tensor(particle.atom_coords)
                    if particle.atom_coords_denoised is not None:
                        atom_coords_denoised_clone = self._safe_clone_tensor(particle.atom_coords_denoised)
                
                point = BoltzTrajectoryPoint(
                    step=step_idx,
                    sigma=sigma,
                    t_hat=t_hat,
                    steering_t=steering_t,
                    interface_score=particle.interface_score,
                    confidence_scores=interface_confidence_scores,
                    atom_coords=atom_coords_clone,
                    atom_coords_denoised=atom_coords_denoised_clone,
                    is_extra_saved_point=is_extra_save
                )
                
                # Store trajectory point
                if i not in self.trajectories:
                    self.trajectories[i] = BoltzParticleTrajectory(particle_id=i)
                
                self.trajectories[i].points.append(point)
                
            except Exception as e:
                logging.error(f"Failed to record step {step_idx} for particle {i}: {e}")
                # Continue with other particles even if one fails
                continue
        
        # Integrate with visualization mixin
        if self.vis_mixin:
            try:
                self.vis_mixin.record_step_extended(
                    step_idx=step_idx, 
                    sigma=sigma,
                    t_hat=t_hat,
                    steering_t=steering_t,
                    particles=particles,
                    confidence_scores=confidence_scores
                )
            except Exception as e:
                logging.warning(f"Visualization recording failed at step {step_idx}: {e}")
    
    def record_resampling_snapshot(self,
                                  step_idx: int,
                                  sigma: float,
                                  t_hat: float,
                                  steering_t: float,
                                  particles: List[BoltzParticleState],
                                  resampling_mapping: Dict[int, int],
                                  confidence_scores: Optional[Dict[str, float]] = None) -> None:
        """Record snapshot before resampling - this is triggered by resample events"""
        
        for i, particle in enumerate(particles):
            # Always save coordinates for resampling points if coordinate saving is enabled
            interface_confidence_scores = confidence_scores
            if self.save_confidence_scores and not interface_confidence_scores and self.confidence_module:
                try:
                    interface_confidence_scores = self._compute_interface_confidence_scores(particle.atom_coords)
                except Exception as e:
                    logging.warning(f"Failed to compute interface confidence scores for resampling at step {step_idx}: {e}")
            
            try:
                # Clone coordinates safely for resampling points
                atom_coords_clone = None
                atom_coords_denoised_clone = None
                
                if self.save_coordinates:
                    if particle.atom_coords is not None:
                        atom_coords_clone = self._safe_clone_tensor(particle.atom_coords)
                    if particle.atom_coords_denoised is not None:
                        atom_coords_denoised_clone = self._safe_clone_tensor(particle.atom_coords_denoised)
                
                # Create resampling point
                resampling_point = BoltzTrajectoryPoint(
                    step=step_idx,
                    sigma=sigma,
                    t_hat=t_hat,
                    steering_t=steering_t,
                    interface_score=particle.interface_score,
                    confidence_scores=interface_confidence_scores,
                    atom_coords=atom_coords_clone,
                    atom_coords_denoised=atom_coords_denoised_clone,
                    is_resampling_point=True
                )
                
                # Store
                if i not in self.trajectories:
                    self.trajectories[i] = BoltzParticleTrajectory(particle_id=i)
                
                self.trajectories[i].points.append(resampling_point)
                
            except Exception as e:
                logging.error(f"Failed to record resampling snapshot for particle {i} at step {step_idx}: {e}")
                continue
        
        # Record resampling event
        self.resampling_events.append({
            'step': step_idx,
            'mapping': resampling_mapping,
            'sigma': sigma,
            't_hat': t_hat,
            'steering_t': steering_t
        })
        
        # Update resampling history for each trajectory
        for new_particle_id, source_particle_id in resampling_mapping.items():
            if new_particle_id in self.trajectories:
                # Record that this particle was resampled from source_particle_id
                self.trajectories[new_particle_id].resampled_from.append(source_particle_id)
        
        # Integrate with visualization mixin
        if self.vis_mixin:
            try:
                self.vis_mixin.record_resampling(step_idx, resampling_mapping)
            except Exception as e:
                logging.warning(f"Visualization resampling recording failed at step {step_idx}: {e}")
        
        logging.info(f"Recorded resampling snapshot at step {step_idx}: {resampling_mapping}")
    
    def _safe_clone_tensor(self, tensor: Tensor) -> Tensor:
        """Safely clone tensor, handling CUDA synchronization"""
        if tensor is None:
            return None
        
        try:
            # Ensure tensor is valid
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logging.warning("Invalid tensor data detected (NaN or inf), skipping clone")
                return None
            
            # Handle CUDA tensors safely
            if tensor.is_cuda:
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                return tensor.detach().cpu().clone()
            else:
                return tensor.detach().clone()
                
        except Exception as e:
            logging.error(f"Failed to clone tensor: {e}")
            return None
    
    def _compute_interface_confidence_scores(self, atom_coords: Tensor) -> Optional[Dict[str, float]]:
        """Compute interface confidence scores (iPTM, mean_iPTM, etc.) with error handling"""
        # Skip confidence calculation in trajectory recorder to avoid parameter mismatch errors
        # The confidence scores are computed in the main diffusion loop and passed directly
        logging.info("Confidence scores will be provided by the main diffusion loop")
        return None
    
    def save_trajectories(self) -> None:
        """Save trajectory data to files"""
        
        # Save coordinates data
        if self.save_coordinates:
            coords_dir = self.output_dir / "coordinates"
            coords_dir.mkdir(exist_ok=True)
            
            for particle_id, trajectory in self.trajectories.items():
                # Save coordinates for points that have them
                coord_points = trajectory.get_points_with_coordinates()
                if coord_points:
                    coords_data = {
                        'particle_id': particle_id,
                        'coordinates': [],
                        'coordinates_denoised': [],
                        'metadata': []
                    }
                    
                    for point in coord_points:
                        # Save original coordinates
                        if point.atom_coords is not None:
                            coords_np = self._tensor_to_numpy(point.atom_coords)
                            coords_data['coordinates'].append(coords_np)
                        else:
                            coords_data['coordinates'].append(None)
                        
                        # Save denoised coordinates
                        if point.atom_coords_denoised is not None:
                            coords_denoised_np = self._tensor_to_numpy(point.atom_coords_denoised)
                            coords_data['coordinates_denoised'].append(coords_denoised_np)
                        else:
                            coords_data['coordinates_denoised'].append(None)
                        
                        coords_data['metadata'].append({
                            'step': point.step,
                            'sigma': point.sigma,
                            't_hat': point.t_hat,
                            'steering_t': point.steering_t,
                            'interface_score': point.interface_score,
                            'is_resampling_point': point.is_resampling_point,
                            'is_extra_saved_point': point.is_extra_saved_point,
                            'confidence_scores': point.confidence_scores
                        })
                    
                    # Save as npz file
                    if coords_data['coordinates']:
                        coord_file = coords_dir / f"particle_{particle_id}_coords.npz"
                        # Filter out None values before saving
                        valid_coords = [c for c in coords_data['coordinates'] if c is not None]
                        valid_coords_denoised = [c for c in coords_data['coordinates_denoised'] if c is not None]
                        
                        if valid_coords:
                            save_data = {
                                'coordinates': np.array(valid_coords),
                                'metadata': coords_data['metadata'],
                                'resampled_from': trajectory.resampled_from  # 包含重采样历史
                            }
                            if valid_coords_denoised:
                                save_data['coordinates_denoised'] = np.array(valid_coords_denoised)
                            
                            np.savez(coord_file, **save_data)
        
        # Save interface confidence scores data
        if self.save_confidence_scores:
            confidence_dir = self.output_dir / "confidence"
            confidence_dir.mkdir(exist_ok=True)
            
            for particle_id, trajectory in self.trajectories.items():
                confidence_data = []
                metadata = []
                
                for point in trajectory.points:
                    if point.confidence_scores is not None:
                        confidence_data.append(point.confidence_scores)
                        metadata.append({
                            'step': point.step,
                            'sigma': point.sigma,
                            't_hat': point.t_hat,
                            'steering_t': point.steering_t,
                            'interface_score': point.interface_score,
                            'is_resampling_point': point.is_resampling_point,
                            'is_extra_saved_point': point.is_extra_saved_point
                        })
                
                if confidence_data:
                    confidence_file = confidence_dir / f"particle_{particle_id}_confidence.json"
                    with open(confidence_file, 'w') as f:
                        json.dump({
                            'confidence_scores': confidence_data,
                            'metadata': metadata
                        }, f, indent=2)
        
        # Save resampling events
        if self.resampling_events:
            resampling_file = self.output_dir / "resampling_events.json"
            with open(resampling_file, 'w') as f:
                json.dump(self.resampling_events, f, indent=2)
        
        # Save trajectory summary
        summary = self.get_trajectory_summary()
        summary_file = self.output_dir / "trajectory_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save trajectory data as CSV for visualization
        if self.vis_mixin:
            try:
                self.vis_mixin.save_trajectory_data_as_csv()
            except Exception as e:
                logging.warning(f"Failed to save visualization CSV data: {e}")
        
        logging.info(f"Saved trajectories for {len(self.trajectories)} particles to {self.output_dir}")
    
    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy array safely"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.array(tensor)
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of recorded trajectories"""
        total_points = sum(len(traj.points) for traj in self.trajectories.values())
        resampling_points = sum(len(traj.get_resampling_points()) for traj in self.trajectories.values())
        extra_saved_points = sum(len(traj.get_extra_saved_points()) for traj in self.trajectories.values())
        coord_points = sum(len(traj.get_points_with_coordinates()) for traj in self.trajectories.values())
        
        return {
            'num_particles': len(self.trajectories),
            'total_points': total_points,
            'resampling_points': resampling_points,
            'extra_saved_points': extra_saved_points,
            'points_with_coordinates': coord_points,
            'num_resampling_events': len(self.resampling_events),
            'save_coordinates': self.save_coordinates,
            'save_confidence_scores': self.save_confidence_scores,
            'extra_save_interval': self.extra_save_interval,
            'output_directory': str(self.output_dir)
        }
    
    def finalize(self) -> None:
        """Finalize recording and save all data"""
        try:
            self.save_trajectories()
            
            summary = self.get_trajectory_summary()
            logging.info(f"BoltzTrajectoryRecorder finalized. "
                        f"Total points: {summary['total_points']}, "
                        f"Resampling points: {summary['resampling_points']}, "
                        f"Extra saved points: {summary['extra_saved_points']}")
        except Exception as e:
            logging.error(f"Failed to finalize Boltz trajectory recording: {e}")


def create_boltz_trajectory_recorder(
    output_dir: Path,
    save_coordinates: bool = True,
    save_confidence_scores: bool = True,
    extra_save_interval: Optional[int] = None,
    confidence_module = None,
    confidence_context: Optional[Dict] = None
) -> BoltzTrajectoryRecorder:
    """Factory function to create a BoltzTrajectoryRecorder instance"""
    return BoltzTrajectoryRecorder(
        output_dir=output_dir,
        save_coordinates=save_coordinates,
        save_confidence_scores=save_confidence_scores,
        extra_save_interval=extra_save_interval,
        confidence_module=confidence_module,
        confidence_context=confidence_context
    )
