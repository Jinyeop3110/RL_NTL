"""
Configuration for Number Token Loss (NTL) integration in PPO training.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from omegaconf import DictConfig


@dataclass
class NTLConfig:
    """Configuration for Number Token Loss integration."""
    
    # Core NTL settings
    enabled: bool = True
    """Whether to enable NTL extraction and computation."""
    
    method: Literal['mse', 'wasserstein'] = 'mse'
    """NTL loss computation method. 'mse' uses Mean Squared Error, 'wasserstein' uses Wasserstein distance."""
    
    weight: float = 0.1
    """Weight for NTL loss when combining with base loss (for loss augmentation)."""
    
    # Extraction settings
    extract_during_generation: bool = False
    """Whether to extract NTL info during generation phase. If False, extract during compute_log_prob."""
    
    extract_during_log_prob: bool = True
    """Whether to extract NTL info during log probability computation."""
    
    # Reward function integration
    pass_to_reward_function: bool = True
    """Whether to pass NTL information to the reward function."""
    
    reward_mode: Literal['final_answer', 'all_tokens', 'hybrid'] = 'final_answer'
    """NTL reward mode:
    - 'final_answer': Only compute NTL for final answer digits
    - 'all_tokens': Compute NTL for all digit tokens in sequence
    - 'hybrid': Use different strategies based on correctness
    """
    
    # Temperature settings for reward computation
    temperature: float = 1.0
    """Temperature parameter (Tau) for exp(-ntl_loss/Tau) reward conversion."""
    
    adaptive_temperature: bool = False  
    """Whether to use adaptive temperature scheduling during training."""
    
    temperature_schedule: Optional[str] = None
    """Temperature schedule string, e.g., 'linear:2.0:0.5' for linear decay from 2.0 to 0.5."""
    
    # Debugging and monitoring
    log_metrics: bool = True
    """Whether to log NTL metrics (loss, accuracy, etc.) during training."""
    
    debug_mode: bool = False
    """Enable debug logging for NTL extraction and computation."""
    
    save_digit_info: bool = False
    """Whether to save detailed digit information for analysis."""
    
    # Performance settings
    cache_digit_mappings: bool = True
    """Whether to cache tokenizer digit mappings for performance."""
    
    batch_ntl_computation: bool = True
    """Whether to compute NTL in batches for efficiency."""
    
    # Compatibility settings
    fallback_on_error: bool = True
    """Whether to fallback to non-NTL behavior if NTL extraction fails."""
    
    require_digits: bool = False
    """Whether to require digit tokens in sequences (raise error if none found)."""


@dataclass 
class NTLTrainerConfig:
    """Extended trainer config with NTL support."""
    
    ntl: NTLConfig = field(default_factory=NTLConfig)
    """NTL-specific configuration."""
    
    use_ntl_trainer: bool = True
    """Whether to use the NTL-enabled trainer instead of standard trainer."""
    
    def validate_ntl_config(self):
        """Validate NTL configuration settings."""
        if self.ntl.enabled:
            if self.ntl.weight < 0.0:
                raise ValueError(f"NTL weight must be non-negative, got {self.ntl.weight}")
            
            if self.ntl.temperature <= 0.0:
                raise ValueError(f"NTL temperature must be positive, got {self.ntl.temperature}")
            
            if not (self.ntl.extract_during_generation or self.ntl.extract_during_log_prob):
                raise ValueError("At least one NTL extraction method must be enabled")
            
            if self.ntl.adaptive_temperature and self.ntl.temperature_schedule is None:
                raise ValueError("Temperature schedule required when adaptive_temperature=True")


def create_ntl_config_from_omegaconf(config: DictConfig) -> NTLTrainerConfig:
    """Create NTL trainer config from OmegaConf configuration."""
    from omegaconf import OmegaConf
    
    # Extract NTL config if present
    ntl_config_dict = config.get('ntl', {})
    
    # Create NTL config
    ntl_config = NTLConfig()
    if ntl_config_dict:
        # Update with provided values
        for key, value in ntl_config_dict.items():
            if hasattr(ntl_config, key):
                setattr(ntl_config, key, value)
            else:
                print(f"Warning: Unknown NTL config key '{key}'")
    
    # Create trainer config
    trainer_config = NTLTrainerConfig(ntl=ntl_config)
    
    # Set use_ntl_trainer based on config
    trainer_config.use_ntl_trainer = config.get('use_ntl_trainer', ntl_config.enabled)
    
    # Validate configuration
    trainer_config.validate_ntl_config()
    
    print(f"[NTL Config] Created NTL config: enabled={ntl_config.enabled}, "
          f"method={ntl_config.method}, reward_mode={ntl_config.reward_mode}")
    
    return trainer_config


def get_default_ntl_config() -> dict:
    """Get default NTL configuration as a dictionary."""
    return {
        'ntl': {
            'enabled': True,
            'method': 'mse',
            'weight': 0.1,
            'extract_during_generation': False,
            'extract_during_log_prob': True,
            'pass_to_reward_function': True,
            'reward_mode': 'final_answer',
            'temperature': 1.0,
            'adaptive_temperature': False,
            'log_metrics': True,
            'debug_mode': False,
            'cache_digit_mappings': True,
            'fallback_on_error': True,
            'require_digits': False,
        },
        'use_ntl_trainer': True
    }


def merge_ntl_config_with_base(base_config: DictConfig, ntl_overrides: dict = None) -> DictConfig:
    """Merge NTL configuration with base training configuration."""
    from omegaconf import OmegaConf
    
    # Get default NTL config
    default_ntl = get_default_ntl_config()
    
    # Apply overrides if provided
    if ntl_overrides:
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(default_ntl, ntl_overrides)
    
    # Merge with base config
    merged_config = OmegaConf.merge(base_config, OmegaConf.create(default_ntl))
    
    return merged_config


# Example usage configurations
EXAMPLE_CONFIGS = {
    'final_answer_mode': {
        'ntl': {
            'enabled': True,
            'method': 'mse',
            'reward_mode': 'final_answer',
            'temperature': 1.0,
            'pass_to_reward_function': True,
        }
    },
    
    'all_tokens_mode': {
        'ntl': {
            'enabled': True,
            'method': 'mse', 
            'reward_mode': 'all_tokens',
            'temperature': 1.0,
            'pass_to_reward_function': True,
        }
    },
    
    'aggressive_ntl': {
        'ntl': {
            'enabled': True,
            'method': 'mse',
            'reward_mode': 'final_answer',
            'temperature': 0.5,  # More sensitive to NTL quality
            'weight': 0.2,
            'log_metrics': True,
        }
    },
    
    'forgiving_ntl': {
        'ntl': {
            'enabled': True,
            'method': 'mse',
            'reward_mode': 'final_answer', 
            'temperature': 2.0,  # More forgiving of NTL errors
            'weight': 0.05,
            'log_metrics': True,
        }
    },
    
    'debug_mode': {
        'ntl': {
            'enabled': True,
            'method': 'mse',
            'reward_mode': 'final_answer',
            'temperature': 1.0,
            'debug_mode': True,
            'save_digit_info': True,
            'log_metrics': True,
        }
    }
}