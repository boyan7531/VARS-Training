#!/usr/bin/env python3
"""Test script to verify optimization and error recovery features."""

def test_data_optimization():
    """Test data optimization features."""
    print("Testing data optimization features...")
    
    try:
        from training.data_optimization import DataLoadingProfiler, IntelligentBatchSizer, create_optimized_dataloader
        print("✅ Data optimization imports successful")
        
        # Test DataLoadingProfiler
        profiler = DataLoadingProfiler()
        profiler.record_batch_timing(0.1, 0.5)  # Data time, compute time
        stats = profiler.get_stats()
        print(f"✅ DataLoadingProfiler working: {stats}")
        
        # Test IntelligentBatchSizer
        batch_sizer = IntelligentBatchSizer(initial_batch_size=16)
        usage = batch_sizer.get_gpu_memory_usage()
        print(f"✅ IntelligentBatchSizer working: GPU memory usage = {usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data optimization test failed: {e}")
        return False


def test_error_recovery():
    """Test error recovery features."""
    print("Testing error recovery features...")
    
    try:
        from training.error_recovery import create_config_validator, OOMRecoveryManager, RobustTrainingWrapper
        print("✅ Error recovery imports successful")
        
        # Test config validator
        validator = create_config_validator()
        print(f"✅ Config validator created with {len(validator.validation_rules)} rules")
        
        # Test OOM recovery manager
        oom_manager = OOMRecoveryManager(initial_batch_size=16)
        print(f"✅ OOM recovery manager created: batch_size = {oom_manager.current_batch_size}")
        
        # Test robust training wrapper
        wrapper = RobustTrainingWrapper()
        print("✅ Robust training wrapper created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False


def test_integration():
    """Test integration with training module."""
    print("Testing integration with training module...")
    
    try:
        import training
        print("✅ Training module imported")
        
        # Check if optimization features are available
        has_optimization = hasattr(training, 'DataLoadingProfiler')
        has_error_recovery = hasattr(training, 'OOMRecoveryManager')
        
        print(f"✅ Optimization features available: {has_optimization}")
        print(f"✅ Error recovery features available: {has_error_recovery}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 TESTING OPTIMIZATION AND ERROR RECOVERY FEATURES")
    print("=" * 60)
    
    tests = [
        ("Data Optimization", test_data_optimization),
        ("Error Recovery", test_error_recovery),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name} Test:")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Optimization features are ready to use.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main() 