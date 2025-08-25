#!/usr/bin/env python3
"""
Validate COCO dataset for YOLO training.

This script uses the dataset validator to check the COCO dataset for issues
that might cause tensor shape mismatches during training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.dataset_validator import DatasetValidator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to validate COCO dataset."""
    # Dataset path
    dataset_path = Path("yolo11_project/datasets/coco")
    
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return 1
    
    logger.info(f"Validating COCO dataset at: {dataset_path}")
    
    # Create validator for COCO dataset (80 classes)
    validator = DatasetValidator(dataset_path)
    
    # Validate dataset
    try:
        summary = validator.validate_dataset(num_classes=80, force=False)
        
        # Print detailed summary
        print("\n" + "="*60)
        print("COCO DATASET VALIDATION RESULTS")
        print("="*60)
        print(f"Total label files:     {summary['total_files']}")
        print(f"Valid files:           {summary['valid_files']}")
        print(f"Invalid files:         {summary['invalid_files']}")
        print(f"Valid percentage:      {(summary['valid_files']/summary['total_files']*100):.2f}%")
        print(f"Total annotations:     {summary['total_annotations']}")
        print(f"Max class index found: {summary['max_class_index_found']}")
        
        # Show details if there are invalid files
        if summary['invalid_files'] > 0:
            print(f"\nInvalid files detected:")
            print("-" * 30)
            for i, file_path in enumerate(summary['invalid_file_list'][:20]):
                print(f"  {i+1:2d}. {file_path}")
            if len(summary['invalid_file_list']) > 20:
                print(f"  ... and {len(summary['invalid_file_list']) - 20} more")
                
            # Check for class index issues specifically
            high_class_files = validator.get_files_with_high_class_indices(80)
            if high_class_files:
                print(f"\nFiles with invalid class indices (>= 80):")
                print("-" * 45)
                for i, file_path in enumerate(high_class_files[:10]):
                    print(f"  {i+1:2d}. {file_path}")
                if len(high_class_files) > 10:
                    print(f"  ... and {len(high_class_files) - 10} more")
        
        # Options for handling invalid files
        if summary['invalid_files'] > 0:
            print(f"\nOptions:")
            print("-" * 10)
            print("1. Run with --delete-invalid flag to remove problematic files")
            print("2. Manually inspect the invalid files listed above")
            print("3. Re-run with --force flag to ignore cache and re-validate")
            
        print(f"\nValidation cache saved to: {validator.cache_file}")
        
        return 0 if summary['invalid_files'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())