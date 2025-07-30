#!/usr/bin/env python3
"""
Dataset validator for YOLO training datasets.

This script validates YOLO dataset annotations to identify:
- Invalid class indices
- Malformed annotation files
- Other issues that could cause training errors

The results are cached for faster subsequent runs.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validate YOLO dataset annotations for potential issues."""
    
    def __init__(self, dataset_path: str, cache_dir: str = None):
        """
        Initialize the dataset validator.
        
        Args:
            dataset_path: Path to the dataset directory
            cache_dir: Directory to store validation cache (default: dataset_path/.cache)
        """
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.dataset_path / '.cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / 'validation_cache.json'
        self.results = {}
        
    def _get_dataset_hash(self) -> str:
        """Generate a hash of the dataset for cache validation."""
        # Get all label files and their modification times
        label_files = list(self.dataset_path.rglob('*.txt'))
        label_files.sort()
        
        hash_input = ""
        for file in label_files[:1000]:  # Limit to first 1000 files for performance
            if file.exists():
                stat = file.stat()
                hash_input += f"{file}:{stat.st_mtime}"
                
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_cache(self) -> bool:
        """Load validation results from cache if available and valid."""
        if not self.cache_file.exists():
            return False
            
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is still valid
            if cache_data.get('dataset_hash') == self._get_dataset_hash():
                self.results = cache_data.get('results', {})
                logger.info(f"Loaded validation results from cache ({len(self.results)} files)")
                return True
            else:
                logger.info("Cache is outdated, will re-validate dataset")
                return False
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self):
        """Save validation results to cache."""
        cache_data = {
            'dataset_hash': self._get_dataset_hash(),
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved validation results to cache ({len(self.results)} files)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _validate_label_file(self, file_path: Path, num_classes: int) -> Dict[str, Any]:
        """
        Validate a single label file.
        
        Args:
            file_path: Path to the label file
            num_classes: Number of expected classes (0 to num_classes-1)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': True,
            'issues': [],
            'max_class_index': -1,
            'total_annotations': 0
        }
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            result['total_annotations'] = len(lines)
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        result['valid'] = False
                        result['issues'].append(f"Line {i+1}: Insufficient data (< 5 values)")
                        continue
                        
                    class_idx = int(parts[0])
                    result['max_class_index'] = max(result['max_class_index'], class_idx)
                    
                    if class_idx < 0:
                        result['valid'] = False
                        result['issues'].append(f"Line {i+1}: Negative class index {class_idx}")
                    elif class_idx >= num_classes:
                        result['valid'] = False
                        result['issues'].append(f"Line {i+1}: Class index {class_idx} >= num_classes {num_classes}")
                    
                    # Check bounding box coordinates (should be 0-1)
                    coords = [float(x) for x in parts[1:5]]
                    for j, coord in enumerate(coords):
                        if coord < 0 or coord > 1:
                            result['valid'] = False
                            result['issues'].append(f"Line {i+1}: Coordinate {j+1} out of range [0,1]: {coord}")
                            
                except ValueError as e:
                    result['valid'] = False
                    result['issues'].append(f"Line {i+1}: Invalid number format: {e}")
                except Exception as e:
                    result['valid'] = False
                    result['issues'].append(f"Line {i+1}: Error parsing: {e}")
                    
        except Exception as e:
            result['valid'] = False
            result['issues'].append(f"File read error: {e}")
            
        return result
    
    def validate_dataset(self, num_classes: int = 80, force: bool = False) -> Dict[str, Any]:
        """
        Validate the entire dataset.
        
        Args:
            num_classes: Number of expected classes (0 to num_classes-1)
            force: Force re-validation even if cache exists
            
        Returns:
            Dictionary with validation summary
        """
        if not force and self._load_cache():
            logger.info("Using cached validation results")
        else:
            logger.info("Starting dataset validation...")
            
            # Find all label files
            label_files = list(self.dataset_path.rglob('*.txt'))
            logger.info(f"Found {len(label_files)} label files to validate")
            
            # Validate each file
            invalid_files = []
            valid_files = []
            total_annotations = 0
            max_class_found = -1
            
            for i, file_path in enumerate(label_files):
                if i % 1000 == 0:
                    logger.info(f"Validated {i}/{len(label_files)} files...")
                    
                rel_path = str(file_path.relative_to(self.dataset_path))
                result = self._validate_label_file(file_path, num_classes)
                self.results[rel_path] = result
                
                if result['valid']:
                    valid_files.append(rel_path)
                else:
                    invalid_files.append(rel_path)
                    logger.warning(f"Invalid file {rel_path}: {result['issues'][:3]}")  # Show first 3 issues
                    
                total_annotations += result['total_annotations']
                max_class_found = max(max_class_found, result['max_class_index'])
            
            # Save results to cache
            self._save_cache()
            
            # Prepare summary
            summary = {
                'total_files': len(label_files),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'total_annotations': total_annotations,
                'max_class_index_found': max_class_found,
                'invalid_file_list': invalid_files[:100]  # Limit the list for readability
            }
            
            return summary
    
    def get_invalid_files(self) -> List[str]:
        """Get list of invalid files from validation results."""
        return [path for path, result in self.results.items() if not result['valid']]
    
    def get_files_with_high_class_indices(self, min_class: int) -> List[str]:
        """Get files with class indices >= min_class."""
        return [path for path, result in self.results.items() 
                if result['max_class_index'] >= min_class]
    
    def delete_invalid_files(self) -> int:
        """Delete invalid label files and their corresponding images."""
        invalid_files = self.get_invalid_files()
        deleted_count = 0
        
        for rel_path in invalid_files:
            label_path = self.dataset_path / rel_path
            
            # Delete label file
            if label_path.exists():
                try:
                    label_path.unlink()
                    logger.info(f"Deleted invalid label file: {rel_path}")
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {rel_path}: {e}")
                    
                # Try to delete corresponding image file
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                for ext in image_extensions:
                    image_path = label_path.with_suffix(ext)
                    if image_path.exists():
                        try:
                            image_path.unlink()
                            logger.info(f"Deleted corresponding image: {image_path.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete {image_path.name}: {e}")
                        break
                        
        return deleted_count

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Validate YOLO dataset annotations')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of expected classes')
    parser.add_argument('--delete-invalid', action='store_true', 
                        help='Delete invalid files and corresponding images')
    parser.add_argument('--force', action='store_true', 
                        help='Force re-validation (ignore cache)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create validator
    validator = DatasetValidator(args.dataset_path)
    
    # Validate dataset
    summary = validator.validate_dataset(args.num_classes, args.force)
    
    # Print summary
    print("\nValidation Summary:")
    print("=" * 50)
    print(f"Total files: {summary['total_files']}")
    print(f"Valid files: {summary['valid_files']}")
    print(f"Invalid files: {summary['invalid_files']}")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Max class index found: {summary['max_class_index_found']}")
    
    if summary['invalid_files'] > 0:
        print(f"\nFirst {min(10, len(summary['invalid_file_list']))} invalid files:")
        for file_path in summary['invalid_file_list'][:10]:
            print(f"  - {file_path}")
    
    # Delete invalid files if requested
    if args.delete_invalid and summary['invalid_files'] > 0:
        deleted_count = validator.delete_invalid_files()
        print(f"\nDeleted {deleted_count} invalid files")
    
    return 0 if summary['invalid_files'] == 0 else 1

if __name__ == '__main__':
    exit(main())