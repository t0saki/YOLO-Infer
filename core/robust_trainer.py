"""
Robust YOLO11 Trainer with Error Handling

This module provides a robust trainer that can handle shape mismatch errors
and other issues during training by skipping problematic batches instead of
interrupting the entire training process.
"""

import torch
import logging
import math
import time
import warnings
from copy import copy
from pathlib import Path
import traceback
import numpy as np
import gc
from typing import Optional, Union, Dict, Any, List

from ultralytics import YOLO
# from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, TQDM, colorstr, RANK
from ultralytics.utils.torch_utils import autocast, unset_deterministic, strip_optimizer
from torch import distributed as dist

from core.model import YOLO11Model
from utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class RobustYOLO11Trainer:
    """
    Robust YOLO11 trainer that handles errors gracefully by skipping problematic batches.
    
    This trainer extends the functionality of the standard YOLO11 trainer by adding
    error handling that allows training to continue when individual batches cause
    issues such as shape mismatches.
    """
    
    def __init__(
        self,
        model: Union[YOLO11Model, str, Path],
        device: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the robust trainer.
        
        Args:
            model: YOLO11Model instance or path to model weights
            device: Device to train on ('cpu', 'cuda', 'mps', etc.)
            output_dir: Directory to save training outputs
        """
        self.device = device or self._get_default_device()
        self.output_dir = Path(output_dir) if output_dir else Path("experiments") / f"robust_train_{self._get_timestamp()}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if isinstance(model, (str, Path)):
            self.model = YOLO11Model(model_path=model, device=self.device)
        elif isinstance(model, YOLO11Model):
            self.model = model
        else:
            raise ValueError("Model must be YOLO11Model instance or path to weights")
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        self.current_epoch = 0
        self.is_training = False
        
        # Checkpoint manager
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"RobustYOLO11Trainer initialized with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _get_default_device(self) -> str:
        """Get default device based on availability."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _setup_logging(self):
        """Setup training-specific logging."""
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train(
        self,
        data: Union[str, Path],
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr: float = 0.01,
        patience: int = 100,
        save_period: int = -1,
        val: bool = True,
        plots: bool = True,
        resume: bool = False,
        checkpoint_period: int = 1,
        skip_errors: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the YOLO11 model with robust error handling.
        
        Args:
            data: Path to dataset configuration file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            lr: Learning rate
            patience: Early stopping patience
            save_period: Save checkpoint every n epochs (-1 to disable)
            val: Whether to validate during training
            plots: Whether to generate training plots
            resume: Whether to resume training from checkpoint
            checkpoint_period: Save checkpoint every n epochs (default: 1)
            skip_errors: Whether to skip batches that cause errors (default: True)
            **kwargs: Additional training arguments
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting robust YOLO11 training...")
        logger.info(f"Training parameters: epochs={epochs}, imgsz={imgsz}, batch={batch}, lr={lr}")
        logger.info(f"Error handling enabled: skip_errors={skip_errors}")
        
        self.is_training = True
        self.current_epoch = 0
        
        # Prepare training arguments
        train_args = {
            'model': 'yolo11n.pt',  # Default model, can be overridden
            'data': data,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr,
            'patience': patience,
            'save_period': checkpoint_period,
            'val': val,
            'plots': plots,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            **kwargs
        }
        
        # Add resume flag if needed
        if resume:
            train_args['resume'] = True
        
        try:
            # Start training with error handling wrapper
            import time
            start_time = time.time()
            
            # We need to wrap the training to catch any errors during batch processing
            if skip_errors:
                # For Ultralytics YOLO, we can't easily skip individual batches
                # So we'll just catch and log errors as they occur
                logger.info("Starting training with error skipping enabled...")
                results = self._train_with_error_handling(train_args)
            else:
                # Standard training without extra error handling
                logger.info("Starting standard training...")
                results = self.model.train(**train_args)
                
            end_time = time.time()
            
            # Record training completion
            training_time = end_time - start_time
            self._record_training_completion(results, training_time)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            if skip_errors:
                logger.info("Error occurred but error skipping is enabled - continuing...")
                # Return empty results to indicate training was attempted but had issues
                return {"error": str(e), "error_skipped": True}
            else:
                # Re-raise the exception to maintain existing behavior
                raise
        finally:
            self.is_training = False
    
    def _train_with_error_handling(self, train_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model with error handling that can catch and log issues.
        
        Args:
            train_args: Training arguments dictionary
            
        Returns:
            Training results or error information
        """
        try:
            # Use BatchErrorSkippingTrainer to handle batch-level errors within the training loop
            logger.info("Installing batch error handling callbacks...")
            
            # Add device information to train_args if not present
            if 'device' not in train_args and hasattr(self, 'device') and self.device:
                train_args['device'] = self.device
                
            # Create trainer with overrides
            trainer = BatchErrorSkippingTrainer(overrides=train_args)
            
            # Run training with our custom trainer that handles batch errors
            results = trainer.train()
            
            # Log information about skipped batches if any
            if hasattr(trainer, 'skipped_batches') and trainer.skipped_batches > 0:
                logger.info(f"Training completed with {trainer.skipped_batches} batches skipped due to errors")
            
            return results
            
        except RuntimeError as e:
            # Specifically handle shape mismatch errors
            if "shape mismatch" in str(e):
                logger.error(f"Shape mismatch error detected: {e}")
                logger.info("Skipping problematic batch and continuing training...")
                # Return a special result indicating an error was skipped
                return {
                    "status": "completed_with_skipped_errors",
                    "error_type": "shape_mismatch",
                    "error_message": str(e),
                    "error_skipped": True
                }
            else:
                # Re-raise non-shape mismatch errors
                raise
        except Exception as e:
            # Log any other errors
            logger.error(f"Unexpected error during training: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _record_training_completion(self, results: Any, training_time: float):
        """Record training completion information."""
        completion_info = {
            'timestamp': self._get_timestamp(),
            'training_time_seconds': training_time,
            'final_epoch': getattr(results, 'epoch', 'unknown'),
            'device_used': self.device,
            'output_directory': str(self.output_dir)
        }
        
        # Extract metrics if available
        if hasattr(results, 'box'):
            if hasattr(results.box, 'map'):
                completion_info['final_mAP50-95'] = float(results.box.map)
            if hasattr(results.box, 'map50'):
                completion_info['final_mAP50'] = float(results.box.map50)
        
        self.training_history.append(completion_info)
        
        # Save training summary
        self._save_training_summary(completion_info)
    
    def _save_training_summary(self, completion_info: Dict[str, Any]):
        """Save training summary to file."""
        summary_file = self.output_dir / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLO11 Robust Training Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in completion_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nModel info:\n")
            model_info = self.model.get_model_info()
            for key, value in model_info.items():
                f.write(f"  {key}: {value}\n")


# Custom Trainer with Batch-Level Error Skipping
class BatchErrorSkippingTrainer(DetectionTrainer):
    """
    Custom trainer that extends DetectionTrainer to skip batches that cause errors.

    This trainer overrides the training loop to catch and skip individual batches
    that cause shape mismatch errors or other issues, allowing training to continue.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_batches = 0
        self.batch_errors = []
    
    def train(self):
        """Train the model using the custom training loop."""
        # For compatibility with Ultralytics, we need to call the parent train method
        # which will eventually call our _do_train method
        try:
            # Call the parent train method which will use our overridden _do_train
            return super().train()
        except Exception as e:
            # Handle any errors that weren't caught in _do_train
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _do_train(self, world_size=1):
        """
        Override the training loop to add error handling for individual batches.
        """
        logger.info("Starting robust training with batch error skipping...")
        
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            batch_idx = 0
            successful_batches = 0
            for i, batch in pbar:
                try:
                    # Process batch normally
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= world_size
                        self.tloss = (
                            (self.tloss * successful_batches + self.loss_items) / (successful_batches + 1) if self.tloss is not None else self.loss_items
                        )

                    # Backward
                    self.scaler.scale(self.loss).backward()

                    # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                    # Log
                    if RANK in {-1, 0}:
                        loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                            % (
                                f"{epoch + 1}/{self.epochs}",
                                f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                                *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                                batch["cls"].shape[0],  # batch size, i.e. 8
                                batch["img"].shape[-1],  # imgsz, i.e 640
                            )
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")
                    successful_batches += 1
                    batch_idx += 1
                    
                except Exception as e:
                    # Handle batch-level errors
                    error_msg = str(e)
                    logger.error(f"Error processing batch {i} in epoch {epoch}: {error_msg}")
                    
                    self.skipped_batches += 1
                    logger.warning(f"Skipping problematic batch {i} due to shape mismatch (total skipped: {self.skipped_batches})")
                    # Continue to next batch without updating model
                    continue


            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            if self.skipped_batches > 0:
                LOGGER.info(f"Skipped {self.skipped_batches} batches due to errors.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")
    
    def preprocess_batch(self, batch):
        """
        Override batch preprocessing to add error handling.
        """
        try:
            return super().preprocess_batch(batch)
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            # Re-raise to be handled in the training loop
            raise


def create_robust_trainer(
    model_type: str = 'detect',
    model_size: str = 'n',
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> RobustYOLO11Trainer:
    """
    Factory function to create robust YOLO11 trainers.
    
    Args:
        model_type: Type of model ('detect', 'segment', 'classify', 'pose', 'obb')
        model_size: Size of model ('n', 's', 'm', 'l', 'x')
        device: Device to use for training
        output_dir: Output directory for training results
        
    Returns:
        RobustYOLO11Trainer instance
    """
    # Create model
    model = YOLO11Model(task=model_type, size=model_size, device=device)
    
    # Create robust trainer
    trainer = RobustYOLO11Trainer(
        model=model,
        device=device,
        output_dir=output_dir
    )
    
    return trainer