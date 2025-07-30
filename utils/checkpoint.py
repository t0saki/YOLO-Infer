"""
Checkpoint utilities for YOLO11 project.

This module provides utilities for saving and loading training checkpoints,
including model weights, optimizer state, and training progress.
"""

import torch
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Checkpoint manager for saving and loading training states.
    
    This class handles saving and loading of:
    - Model weights
    - Optimizer state
    - Scheduler state
    - Training progress (epoch, step, metrics)
    - Random state for reproducibility
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        best_metric: Optional[float] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
            best_metric: Best metric value so far
            filename: Checkpoint filename (default: checkpoint_epoch_{epoch}.pt)
            
        Returns:
            Path to saved checkpoint
        """
        state = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics or {},
            'best_metric': best_metric,
            'model_state_dict': model.state_dict()
        }
        
        # Save optimizer state if provided
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
            
        # Save scheduler state if provided
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save random states for reproducibility
        state['torch_rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda_rng_state'] = torch.cuda.get_rng_state()
            
        # Generate filename
        if filename is None:
            if epoch is not None:
                filename = f'checkpoint_epoch_{epoch}.pt'
            else:
                filename = 'checkpoint.pt'
                
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load tensors to
            
        Returns:
            Checkpoint state dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)
            
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Restore random states
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'torch_cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
            
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
        return checkpoint
    
    def save_best_model(
        self,
        model: torch.nn.Module,
        metric_value: float,
        metric_name: str = 'metric'
    ) -> Path:
        """
        Save best model based on metric.
        
        Args:
            model: Model to save
            metric_value: Metric value
            metric_name: Metric name
            
        Returns:
            Path to saved best model
        """
        filename = f'best_{metric_name}_{metric_value:.4f}.pt'
        checkpoint_path = self.checkpoint_dir / filename
        
        state = {
            'model_state_dict': model.state_dict(),
            'metric_value': metric_value,
            'metric_name': metric_name
        }
        
        torch.save(state, checkpoint_path)
        logger.info(f"Best model saved to: {checkpoint_path}")
        
        return checkpoint_path
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the latest checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoint_files:
            # Try generic checkpoint name
            generic_checkpoint = self.checkpoint_dir / 'checkpoint.pt'
            if generic_checkpoint.exists():
                return generic_checkpoint
            return None
            
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """
        Get the best checkpoint file based on metric.
        
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        # This would require parsing filenames or keeping track of best metrics
        # For now, we'll return the latest checkpoint as a fallback
        return self.get_latest_checkpoint()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with their information.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint*.pt'))
        checkpoints = []
        
        for checkpoint_file in checkpoint_files:
            try:
                stat = checkpoint_file.stat()
                checkpoint_info = {
                    'path': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime),
                    'is_latest': False
                }
                
                # Try to extract epoch from filename
                if 'epoch_' in checkpoint_file.name:
                    try:
                        epoch = int(checkpoint_file.name.split('epoch_')[1].split('.')[0])
                        checkpoint_info['epoch'] = epoch
                    except:
                        pass
                
                checkpoints.append(checkpoint_info)
            except Exception as e:
                logger.warning(f"Failed to get info for checkpoint {checkpoint_file}: {e}")
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
        
        # Mark latest checkpoint
        if checkpoints:
            checkpoints[0]['is_latest'] = True
            
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get detailed information about a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            # Load checkpoint header only (not the full state dict)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'path': str(checkpoint_path),
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'modified_time': datetime.fromtimestamp(checkpoint_path.stat().st_mtime),
                'epoch': checkpoint.get('epoch', None),
                'metrics': checkpoint.get('metrics', {}),
                'has_optimizer_state': 'optimizer_state_dict' in checkpoint,
                'has_scheduler_state': 'scheduler_state_dict' in checkpoint
            }
            
            return info
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            raise
    
    def cleanup_checkpoints(self, keep_last_n: int = 5):
        """
        Clean up old checkpoints, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if len(checkpoint_files) <= keep_last_n:
            return
            
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        for checkpoint_file in checkpoint_files[:-keep_last_n]:
            try:
                checkpoint_file.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
    
    def remove_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Remove a specific checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file to remove
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"Removed checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint_path}: {e}")
                raise
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")


def save_training_state(
    checkpoint_dir: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save complete training state.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        epoch: Current epoch
        step: Current step
        metrics: Current metrics
        config: Training configuration
        
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    
    # Add config to metrics for saving
    if config is not None:
        if metrics is None:
            metrics = {}
        metrics['config'] = config
        
    return manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=step,
        metrics=metrics
    )


def load_training_state(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load complete training state.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Checkpoint state dictionary
    """
    manager = CheckpointManager(Path(checkpoint_path).parent)
    return manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )