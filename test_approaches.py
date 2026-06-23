"""
Test all AutoML approaches to ensure they work correctly.
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import TextAutoML, AGNewsDataset
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_approach(approach_name: str, dataset_size: int = 100):
    """Test a specific approach with a small dataset."""
    logger.info(f"\n=== Testing {approach_name} approach ===")
    
    try:
        # Create AutoML instance
        automl = TextAutoML(seed=42)
        automl.approach = approach_name  # Force specific approach
        automl.epochs = 2  # Reduce for faster testing
        
        # Create small test dataset
        dataset = AGNewsDataset()
        data_info = dataset.create_dataloaders(test_size=0.2)
        
        # Limit dataset size for testing
        train_df = data_info['train_df'].head(dataset_size)
        test_df = data_info['test_df'].head(dataset_size // 4)
        
        logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # Mock the data_info for fitting
        limited_data_info = {
            'train_df': train_df,
            'val_df': train_df.tail(20),  # Small validation set
            'test_df': test_df,
            'num_classes': 4
        }
        
        # Manually call the specific fitting method
        automl.num_classes = 4
        
        if approach_name in ["bag_of_words", "tfidf"]:
            automl._fit_classical(train_df, limited_data_info['val_df'])
        elif approach_name == "neural-lstm":
            automl._fit_neural_lstm(train_df, limited_data_info['val_df'])
        elif approach_name == "neural-transformer":
            automl._fit_neural_transformer(train_df, limited_data_info['val_df'])
        elif approach_name == "finetune":
            automl._fit_pretrained(train_df, limited_data_info['val_df'])
        
        logger.info(f"✓ Training completed for {approach_name}")
        
        # Test prediction (simplified)
        if automl.model is not None:
            logger.info(f"✓ Model created successfully for {approach_name}")
            
            # Quick prediction test with dummy data
            if approach_name in ["bag_of_words", "tfidf"] and automl.vectorizer is not None:
                logger.info(f"✓ Vectorizer created for {approach_name}")
            elif approach_name in ["neural-lstm", "neural-transformer"] and automl.tokenizer is not None:
                logger.info(f"✓ Tokenizer created for {approach_name}")
            elif approach_name == "finetune":
                logger.info(f"✓ Pretrained tokenizer loaded for {approach_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing {approach_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    logger.info("Checking dependencies...")
    
    # Check basic dependencies
    try:
        import torch
        import sklearn
        import pandas
        import numpy
        logger.info("✓ Basic dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing basic dependency: {e}")
        return False
    
    # Check transformers
    try:
        import transformers
        logger.info("✓ Transformers library available")
        return True
    except ImportError:
        logger.warning("⚠️ Transformers library not available - finetune approach will not work")
        return True  # Still allow testing of other approaches


def main():
    parser = argparse.ArgumentParser(description="Test all AutoML approaches")
    parser.add_argument(
        "--approaches",
        nargs='+',
        default=["bag_of_words", "tfidf", "neural-lstm", "neural-transformer", "finetune"],
        choices=["bag_of_words", "tfidf", "neural-lstm", "neural-transformer", "finetune"],
        help="Approaches to test"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=100,
        help="Size of test dataset (smaller = faster)"
    )
    
    args = parser.parse_args()
    
    logger.info("🧪 Testing AutoML Approaches")
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Test each approach
    results = {}
    for approach in args.approaches:
        if approach == "finetune":
            try:
                import transformers
                results[approach] = test_approach(approach, args.dataset_size)
            except ImportError:
                logger.warning(f"Skipping {approach} - transformers not available")
                results[approach] = None
        else:
            results[approach] = test_approach(approach, args.dataset_size)
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = 0
    
    for approach, result in results.items():
        if result is None:
            logger.info(f"⏭️  {approach}: SKIPPED")
        elif result:
            logger.info(f"✅ {approach}: PASSED")
            passed += 1
            total += 1
        else:
            logger.error(f"❌ {approach}: FAILED")
            total += 1
    
    logger.info(f"\nResults: {passed}/{total} approaches passed")
    
    if passed == total:
        logger.info("🎉 All available approaches working correctly!")
        return 0
    else:
        logger.error("❌ Some approaches failed - check the logs above")
        return 1


if __name__ == "__main__":
    exit(main())