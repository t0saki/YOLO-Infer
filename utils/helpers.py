"""
Helper utilities for YOLO11 project.

This module provides common helper functions used throughout the project.
"""

import torch
import psutil
import GPUtil
import platform
import time
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
import json
import yaml

logger = logging.getLogger(__name__)


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    }
    
    # CUDA information
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['cuda_device_count'] = torch.cuda.device_count()
        
        # GPU information
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                })
            info['gpus'] = gpu_info
        except Exception as e:
            logger.warning(f"Could not get GPU information: {e}")
            info['gpus'] = []
    
    return info


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size and parameter count.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model size information
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            non_trainable_params += param.numel()
    
    # Calculate size in different units
    param_size_bytes = total_params * 4  # Assuming float32
    param_size_mb = param_size_bytes / (1024 * 1024)
    param_size_gb = param_size_mb / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'size_bytes': param_size_bytes,
        'size_mb': param_size_mb,
        'size_gb': param_size_gb
    }


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.2f}s"


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes in human-readable format.
    
    Args:
        bytes_size: Size in bytes
    
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return config or {}
        
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise


def create_experiment_dir(
    base_dir: Union[str, Path] = "experiments",
    experiment_name: Optional[str] = None
) -> Path:
    """
    Create a unique experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
    
    Returns:
        Path to created experiment directory
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        # Generate timestamp-based name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"
    
    # Ensure unique directory name
    exp_dir = base_dir / experiment_name
    counter = 1
    while exp_dir.exists():
        exp_dir = base_dir / f"{experiment_name}_{counter}"
        counter += 1
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    console_output: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        console_output: Whether to output to console
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(level=numeric_level, handlers=[])
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    # Add file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


class Timer:
    """Simple timer context manager and class."""
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize timer.
        
        Args:
            name: Timer name for logging
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        return self.elapsed_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        logger.info(f"{self.name} completed in {format_time(self.elapsed_time)}")


class ProgressTracker:
    """Progress tracking utility."""
    
    def __init__(self, total: int, name: str = "Progress"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            name: Progress tracker name
        """
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        
        # Calculate progress
        progress = self.current / self.total
        elapsed = time.time() - self.start_time
        
        if progress > 0:
            eta = elapsed / progress - elapsed
            logger.info(
                f"{self.name}: {progress:.1%} ({self.current}/{self.total}) "
                f"- Elapsed: {format_time(elapsed)} "
                f"- ETA: {format_time(eta)}"
            )
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name} completed in {format_time(elapsed)}")


def validate_model_path(model_path: Union[str, Path]) -> Path:
    """
    Validate and normalize model path.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file is not a valid model file
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check file extension
    valid_extensions = {'.pt', '.pth', '.onnx', '.trt', '.engine'}
    if model_path.suffix.lower() not in valid_extensions:
        logger.warning(f"Unexpected model file extension: {model_path.suffix}")
    
    return model_path


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
    
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config:
            _deep_update(merged, config)
    
    return merged


def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """
    Deep update a dictionary with another dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_directory(
    directory: Union[str, Path],
    keep_files: Optional[List[str]] = None,
    keep_extensions: Optional[List[str]] = None
) -> None:
    """
    Clean directory contents while optionally keeping specific files.
    
    Args:
        directory: Directory to clean
        keep_files: List of filenames to keep
        keep_extensions: List of file extensions to keep
    """
    directory = Path(directory)
    
    if not directory.exists():
        return
    
    keep_files = keep_files or []
    keep_extensions = keep_extensions or []
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            # Check if file should be kept
            should_keep = False
            
            # Check filename
            if file_path.name in keep_files:
                should_keep = True
            
            # Check extension
            if file_path.suffix.lower() in [ext.lower() for ext in keep_extensions]:
                should_keep = True
            
            # Remove file if not keeping
            if not should_keep:
                try:
                    file_path.unlink()
                    logger.debug(f"Removed file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")


def find_files_by_pattern(
    directory: Union[str, Path],
    pattern: str,
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: File pattern (e.g., '*.py', '**/*.txt')
        recursive: Whether to search recursively
    
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """
    Get hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
    
    Returns:
        Hexadecimal hash string
    """
    import hashlib
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
    """
    Compare two files for equality.
    
    Args:
        file1: First file path
        file2: Second file path
    
    Returns:
        True if files are identical, False otherwise
    """
    try:
        hash1 = get_file_hash(file1)
        hash2 = get_file_hash(file2)
        return hash1 == hash2
    except Exception as e:
        logger.error(f"Error comparing files: {e}")
        return False


def backup_file(
    file_path: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    timestamp: bool = True
) -> Path:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)
        timestamp: Whether to add timestamp to backup filename
    
    Returns:
        Path to backup file
    """
    import shutil
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine backup directory
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename
    backup_name = file_path.name
    if timestamp:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp_str}{file_path.suffix}"
    
    backup_path = backup_dir / backup_name
    
    # Copy file
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backup created: {backup_path}")
    
    return backup_path


def download_file(
    url: str,
    destination: Union[str, Path],
    chunk_size: int = 8192,
    show_progress: bool = True
) -> Path:
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        destination: Destination file path
        chunk_size: Chunk size for downloading
        show_progress: Whether to show download progress
    
    Returns:
        Path to downloaded file
    """
    import requests
    from tqdm import tqdm
    
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        if show_progress and total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    
    logger.info(f"Downloaded: {destination}")
    return destination


def check_dependencies(required_packages: List[str]) -> Dict[str, bool]:
    """
    Check if required packages are installed.
    
    Args:
        required_packages: List of package names to check
    
    Returns:
        Dictionary mapping package names to availability status
    """
    import importlib
    
    results = {}
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary with system information
    """
    import socket
    
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': {}
    }
    
    # Get disk usage for each mounted drive
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            info['disk_usage'][partition.device] = {
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': (usage.used / usage.total) * 100
            }
        except PermissionError:
            continue
    
    return info


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.history = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        import threading
        
        self.monitoring = True
        self.history = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Get CPU and memory usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    
                    # Get GPU usage if available
                    gpu_usage = []
                    try:
                        gpus = GPUtil.getGPUs()
                        for gpu in gpus:
                            gpu_usage.append({
                                'id': gpu.id,
                                'load': gpu.load * 100,
                                'memory_used': gpu.memoryUsed,
                                'memory_total': gpu.memoryTotal,
                                'temperature': gpu.temperature
                            })
                    except:
                        pass
                    
                    # Record data point
                    data_point = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used': memory.used,
                        'memory_total': memory.total,
                        'gpu_usage': gpu_usage
                    }
                    
                    self.history.append(data_point)
                    
                    # Limit history to last 1000 points
                    if len(self.history) > 1000:
                        self.history.pop(0)
                    
                    time.sleep(self.interval)
                    
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        logger.info("Resource monitoring stopped")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if not self.history:
            return {}
        return self.history[-1]
    
    def get_average_usage(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get average resource usage."""
        if not self.history:
            return {}
        
        data = self.history[-last_n:] if last_n else self.history
        
        if not data:
            return {}
        
        avg_cpu = sum(d['cpu_percent'] for d in data) / len(data)
        avg_memory = sum(d['memory_percent'] for d in data) / len(data)
        
        result = {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory
        }
        
        # Add GPU averages if available
        if data[0].get('gpu_usage'):
            for i, gpu in enumerate(data[0]['gpu_usage']):
                gpu_loads = [d['gpu_usage'][i]['load'] for d in data if i < len(d['gpu_usage'])]
                if gpu_loads:
                    result[f'avg_gpu_{i}_load'] = sum(gpu_loads) / len(gpu_loads)
        
        return result
    
    def save_history(self, file_path: Union[str, Path]):
        """Save monitoring history to file."""
        import json
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Resource monitoring history saved to: {file_path}")