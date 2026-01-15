"""
Factory functions for creating PPO trainers with optional NTL support.
"""

from typing import Optional, Dict, Any
from omegaconf import DictConfig

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer_ntl import RayPPOTrainerNTL
from verl.trainer.config.ntl_config import create_ntl_config_from_omegaconf, merge_ntl_config_with_base


def create_ppo_trainer(config: DictConfig, 
                      tokenizer=None, 
                      processor=None,
                      use_ntl: Optional[bool] = None,
                      ntl_overrides: Optional[Dict[str, Any]] = None) -> RayPPOTrainer:
    """
    Factory function to create the appropriate PPO trainer based on configuration.
    
    Args:
        config: Base training configuration
        tokenizer: HuggingFace tokenizer
        processor: Optional processor for multimodal models
        use_ntl: Override whether to use NTL trainer (if None, uses config setting)
        ntl_overrides: Optional overrides for NTL configuration
        
    Returns:
        PPO trainer instance (either standard or NTL-enabled)
    """
    # Merge NTL configuration if needed
    if ntl_overrides or not hasattr(config, 'ntl'):
        config = merge_ntl_config_with_base(config, ntl_overrides)
    
    # Create NTL trainer config
    ntl_trainer_config = create_ntl_config_from_omegaconf(config)
    
    # Determine whether to use NTL trainer
    should_use_ntl = use_ntl if use_ntl is not None else ntl_trainer_config.use_ntl_trainer
    
    if should_use_ntl and ntl_trainer_config.ntl.enabled:
        print("[Trainer Factory] Creating NTL-enabled PPO trainer")
        
        # Create NTL trainer with additional parameters
        trainer = RayPPOTrainerNTL(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            ntl_enabled=ntl_trainer_config.ntl.enabled,
            ntl_method=ntl_trainer_config.ntl.method,
            ntl_extract_during_generation=ntl_trainer_config.ntl.extract_during_generation
        )
        
        # Store NTL config for later use
        trainer.ntl_config = ntl_trainer_config.ntl
        
    else:
        print("[Trainer Factory] Creating standard PPO trainer")
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer, 
            processor=processor
        )
        
        # Add empty NTL config for consistency
        trainer.ntl_config = None
    
    return trainer


def create_ntl_trainer(config: DictConfig,
                      tokenizer=None,
                      processor=None,
                      **ntl_kwargs) -> RayPPOTrainerNTL:
    """
    Convenience function to create NTL-enabled PPO trainer with explicit NTL parameters.
    
    Args:
        config: Base training configuration
        tokenizer: HuggingFace tokenizer
        processor: Optional processor for multimodal models
        **ntl_kwargs: Additional NTL configuration parameters
        
    Returns:
        NTL-enabled PPO trainer instance
    """
    # Ensure NTL is enabled in config
    ntl_overrides = {'ntl': {'enabled': True}}
    
    # Add any additional NTL parameters
    if ntl_kwargs:
        ntl_overrides['ntl'].update(ntl_kwargs)
    
    # Merge configuration
    config = merge_ntl_config_with_base(config, ntl_overrides)
    
    # Create NTL trainer config
    ntl_trainer_config = create_ntl_config_from_omegaconf(config)
    
    print(f"[Trainer Factory] Creating NTL trainer with method={ntl_trainer_config.ntl.method}, "
          f"reward_mode={ntl_trainer_config.ntl.reward_mode}")
    
    # Create trainer
    trainer = RayPPOTrainerNTL(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        ntl_enabled=True,
        ntl_method=ntl_trainer_config.ntl.method,
        ntl_extract_during_generation=ntl_trainer_config.ntl.extract_during_generation
    )
    
    # Store configuration
    trainer.ntl_config = ntl_trainer_config.ntl
    
    return trainer


def create_standard_trainer(config: DictConfig,
                          tokenizer=None,
                          processor=None) -> RayPPOTrainer:
    """
    Convenience function to create standard (non-NTL) PPO trainer.
    
    Args:
        config: Training configuration
        tokenizer: HuggingFace tokenizer
        processor: Optional processor for multimodal models
        
    Returns:
        Standard PPO trainer instance
    """
    print("[Trainer Factory] Creating standard PPO trainer")
    
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor
    )
    
    trainer.ntl_config = None
    return trainer


def get_trainer_info(trainer: RayPPOTrainer) -> Dict[str, Any]:
    """
    Get information about the trainer configuration.
    
    Args:
        trainer: PPO trainer instance
        
    Returns:
        Dict with trainer information
    """
    info = {
        'trainer_type': type(trainer).__name__,
        'ntl_enabled': isinstance(trainer, RayPPOTrainerNTL),
        'has_ntl_config': hasattr(trainer, 'ntl_config') and trainer.ntl_config is not None
    }
    
    if info['has_ntl_config']:
        ntl_config = trainer.ntl_config
        info.update({
            'ntl_method': ntl_config.method,
            'ntl_reward_mode': ntl_config.reward_mode,
            'ntl_temperature': ntl_config.temperature,
            'ntl_weight': ntl_config.weight,
        })
    
    return info


# Pre-configured trainer creation functions
def create_final_answer_ntl_trainer(config: DictConfig, tokenizer=None, processor=None, temperature: float = 1.0):
    """Create trainer for final answer NTL mode."""
    return create_ntl_trainer(
        config, tokenizer, processor,
        reward_mode='final_answer',
        temperature=temperature,
        method='mse'
    )


def create_all_tokens_ntl_trainer(config: DictConfig, tokenizer=None, processor=None, temperature: float = 1.0):
    """Create trainer for all tokens NTL mode.""" 
    return create_ntl_trainer(
        config, tokenizer, processor,
        reward_mode='all_tokens',
        temperature=temperature,
        method='mse'
    )


def create_aggressive_ntl_trainer(config: DictConfig, tokenizer=None, processor=None):
    """Create trainer with aggressive (sensitive) NTL settings."""
    return create_ntl_trainer(
        config, tokenizer, processor,
        reward_mode='final_answer',
        temperature=0.5,  # More sensitive
        weight=0.2,       # Higher weight
        method='mse'
    )


def create_forgiving_ntl_trainer(config: DictConfig, tokenizer=None, processor=None):
    """Create trainer with forgiving (less sensitive) NTL settings."""
    return create_ntl_trainer(
        config, tokenizer, processor,
        reward_mode='final_answer', 
        temperature=2.0,  # Less sensitive
        weight=0.05,      # Lower weight
        method='mse'
    )


# Configuration validation
def validate_trainer_config(config: DictConfig, use_ntl: bool = False) -> bool:
    """
    Validate trainer configuration.
    
    Args:
        config: Training configuration to validate
        use_ntl: Whether NTL trainer will be used
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if use_ntl:
        if not hasattr(config, 'ntl'):
            raise ValueError("NTL configuration missing when use_ntl=True")
        
        ntl_trainer_config = create_ntl_config_from_omegaconf(config)
        ntl_trainer_config.validate_ntl_config()
    
    # Add other validation logic here
    return True


# Example usage function
def create_trainer_from_args(config: DictConfig, 
                           tokenizer=None,
                           processor=None,
                           trainer_type: str = 'auto',
                           **kwargs) -> RayPPOTrainer:
    """
    Create trainer based on string specification.
    
    Args:
        config: Training configuration
        tokenizer: HuggingFace tokenizer
        processor: Optional processor
        trainer_type: Type of trainer to create
            - 'auto': Auto-detect from config
            - 'standard': Standard PPO trainer
            - 'ntl': NTL-enabled trainer
            - 'final_answer': Final answer NTL mode
            - 'all_tokens': All tokens NTL mode
            - 'aggressive': Aggressive NTL settings
            - 'forgiving': Forgiving NTL settings
        **kwargs: Additional arguments
        
    Returns:
        PPO trainer instance
    """
    if trainer_type == 'auto':
        return create_ppo_trainer(config, tokenizer, processor, **kwargs)
    elif trainer_type == 'standard':
        return create_standard_trainer(config, tokenizer, processor)
    elif trainer_type == 'ntl':
        return create_ntl_trainer(config, tokenizer, processor, **kwargs)
    elif trainer_type == 'final_answer':
        return create_final_answer_ntl_trainer(config, tokenizer, processor, **kwargs)
    elif trainer_type == 'all_tokens':
        return create_all_tokens_ntl_trainer(config, tokenizer, processor, **kwargs)
    elif trainer_type == 'aggressive':
        return create_aggressive_ntl_trainer(config, tokenizer, processor)
    elif trainer_type == 'forgiving':
        return create_forgiving_ntl_trainer(config, tokenizer, processor)
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")