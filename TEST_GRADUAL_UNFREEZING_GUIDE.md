# Test Scripts for Gradual Unfreezing Issues

This directory contains test scripts to diagnose and reproduce the gradual unfreezing backbone errors that were occurring during training.

## Test Scripts

### 1. `test_gradual_unfreezing_errors.py` - Comprehensive Test
**Purpose**: Simulates the complete gradual unfreezing process that occurs during training.

**What it does**:
- Creates an MViT model with early gradual unfreezing manager
- Tests each epoch from 0 to 10, unfreezing blocks progressively
- Runs multiple forward/backward passes per epoch to catch intermittent issues
- Tests memory usage and gradient computation
- Provides detailed analysis of when and why errors occur

**Usage**:
```bash
python test_gradual_unfreezing_errors.py
```

**Key Features**:
- ✅ Comprehensive epoch-by-epoch testing
- ✅ Multiple iterations per epoch to catch intermittent issues
- ✅ Memory stress testing
- ✅ Gradient computation validation
- ✅ Automatic stabilization testing
- ✅ Detailed failure analysis

### 2. `test_block15_unfreezing.py` - Focused Test
**Purpose**: Specifically targets the block_15 unfreezing scenario that was causing issues in epoch 3.

**What it does**:
- Creates an MViT model and freezes the entire backbone
- Tests forward passes with fully frozen backbone
- Specifically unfreezes block_15 (the problematic block from epoch 3)
- Stress tests the model after block_15 unfreezing
- Tests gradient computation and memory usage

**Usage**:
```bash
python test_block15_unfreezing.py
```

**Key Features**:
- 🎯 Focused on the specific problematic scenario
- 🔄 Stress testing with multiple iterations
- 🧠 Memory stress testing with different batch sizes
- 📈 Gradient computation validation
- 🔧 Automatic model stabilization testing

## What to Look For

### Expected Behavior (After Fixes)
- ✅ **High success rates** (90%+ for both frozen and unfrozen states)
- ✅ **Automatic recovery** messages like "Successfully recovered from RuntimeError with model stabilization"
- ✅ **Stabilization messages** like "Model stabilized after unfreezing"
- ✅ **No NaN gradients** or outputs
- ✅ **Consistent performance** across multiple iterations

### Signs of Issues
- ❌ **Low success rates** after unfreezing
- ❌ **"Unexpected backbone error:"** messages with empty content
- ❌ **Silent RuntimeError** exceptions
- ❌ **NaN gradients** or outputs
- ❌ **Memory allocation errors**
- ❌ **Inconsistent failures** across iterations

## Running on GPU

Both scripts are designed to run on GPU and will automatically:
- Detect CUDA availability
- Move models to GPU
- Use mixed precision (autocast)
- Monitor GPU memory usage
- Clear GPU cache between tests

**To run on your cloud GPU**:
```bash
# Make sure you're in the VARS-Training directory
cd /path/to/VARS-Training

# Run the comprehensive test
python test_gradual_unfreezing_errors.py

# Or run the focused block_15 test
python test_block15_unfreezing.py
```

## Interpreting Results

### Success Case Example:
```
✅ Frozen backbone success rate: 100.0%
🔓 After block_15 unfreezing success rate: 100.0%
📈 Gradient computation: ✅ Success
🎉 No failures detected after block_15 unfreezing!
```

### Problem Case Example:
```
✅ Frozen backbone success rate: 100.0%
🔓 After block_15 unfreezing success rate: 60.0%
📈 Gradient computation: ❌ Failed
🚨 4 failures detected after block_15 unfreezing:
  - Iteration 3: RuntimeError: 
  - Iteration 7: RuntimeError: 
```

## Debug Mode

Both scripts automatically enable debug mode for the backbone processor, which provides detailed logging when errors occur. You'll see messages like:

```
Debug mode for silent backbone errors: enabled
Backbone input: shape=torch.Size([4, 3, 16, 224, 224]), device=cuda:0, dtype=torch.float32, contiguous=True
Successfully recovered from RuntimeError with model stabilization
```

## Troubleshooting

### If tests fail to run:
1. **Check dependencies**: Make sure all required packages are installed
2. **Check GPU memory**: Ensure sufficient GPU memory is available
3. **Check model loading**: Verify the model can be created successfully

### If you see import errors:
```bash
# Make sure you're in the correct directory
cd /path/to/VARS-Training

# Check if the model module can be imported
python -c "from model.unified_model import create_unified_model; print('Import successful')"
```

### If you want more detailed logging:
Edit the logging level in the scripts:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

## Expected Timeline

- **Comprehensive test**: ~5-10 minutes (tests 10 epochs with multiple iterations each)
- **Focused test**: ~2-5 minutes (focused on block_15 scenario)

Both tests will provide real-time progress updates and final summaries.

## Next Steps

Based on the test results:

1. **If tests pass**: The gradual unfreezing stability fixes are working correctly
2. **If tests fail**: The output will help identify:
   - Which specific blocks cause issues
   - Whether stabilization helps
   - What error types are occurring
   - Memory usage patterns

The detailed logs and failure analysis will guide further debugging and fixes if needed. 