# Base data structures
from .base import (
    # PotentialType,
    BoltzParticleState,
    BoltzTrajectoryPoint,
    BoltzParticleTrajectory,
    BoltzInterfaceSteeringConfig,
    should_resample,
    should_apply_interface_guidance
)

# Trajectory recording and visualization
from .vis_mixin import BoltzVisualizationMixin
from .trajectory_recorder import BoltzTrajectoryRecorder, create_boltz_trajectory_recorder

# Physical steering uses boltz's built-in system directly
# No particle filter needed

# Convenience functions
def create_default_config(**kwargs) -> BoltzInterfaceSteeringConfig:
    """Create a default steering configuration with optional overrides"""
    return BoltzInterfaceSteeringConfig(**kwargs)

# Physical scoring removed - use boltz's built-in system directly

# Export main classes for easy import
__all__ = [
    # Base classes
    # "PotentialType",
    "BoltzParticleState", 
    "BoltzTrajectoryPoint",
    "BoltzParticleTrajectory",
    "BoltzInterfaceSteeringConfig",
    "should_resample",
    "should_apply_interface_guidance",
    
    # Trajectory components
    "BoltzVisualizationMixin",
    "BoltzTrajectoryRecorder", 
    "create_boltz_trajectory_recorder",
    
    # Configuration helpers
    "create_default_config",
]
