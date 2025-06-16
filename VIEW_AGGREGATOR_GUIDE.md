# View Aggregator Guide

The VARS-Training project now supports multiple view aggregation strategies for multi-view video analysis.

## Overview

The view aggregator is responsible for combining features from multiple camera views into a single representation for classification. We support three types:

1. **MLP Aggregator** (`mlp`) - Original attention-based MLP aggregation
2. **Transformer Aggregator** (`transformer`) - **[RECOMMENDED]** Self-attention based aggregation with CLS token
3. **MoE Aggregator** (`moe`) - Mixture of Experts for future experiments

## Quick Start

### Using Transformer Aggregator (Default)

```bash
python train_lightning.py \
    --aggregator_type transformer \
    --agg_heads 2 \
    --agg_layers 1 \
    --backbone_type mvit
```

### Comparing Different Aggregators

```bash
# MLP (original)
python train_lightning.py --aggregator_type mlp

# Transformer (recommended) 
python train_lightning.py --aggregator_type transformer --agg_heads 2 --agg_layers 1

# MoE (experimental)
python train_lightning.py --aggregator_type moe
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aggregator_type` | `transformer` | Aggregator type: `mlp`, `transformer`, `moe` |
| `--agg_heads` | `2` | Number of attention heads (transformer only) |
| `--agg_layers` | `1` | Number of transformer encoder layers |
| `--max_views` | `8` | Maximum number of views to handle |

## Hyperparameter Recommendations

### Transformer Aggregator
- **Heads**: Try `{1, 2, 4}` - Start with 2
- **Layers**: Try `{1, 2}` - Start with 1 for efficiency
- **Max Views**: Set based on your dataset (typically 4-8)

### Performance vs. Efficiency
- `agg_heads=2, agg_layers=1`: **Balanced** (recommended)
- `agg_heads=4, agg_layers=2`: **Higher capacity** (if overfitting is not an issue)
- `agg_heads=1, agg_layers=1`: **Most efficient** (for resource-constrained training)

## Architecture Details

### Transformer Aggregator
- Uses learnable CLS token for aggregation
- Positional embeddings for view-specific information
- Self-attention mechanism models inter-view relationships
- Layer normalization for training stability

### Benefits for Sports Video Analysis
- **Inter-view relationships**: Understands how different camera angles complement each other
- **Dynamic attention**: View importance is contextual based on the specific action
- **Robust to variable views**: CLS token + masking handles different numbers of views naturally

## Checkpoint Compatibility

⚠️ **Important**: Checkpoints trained with different aggregator types are **not compatible**.

### Loading Existing Checkpoints

```bash
# For checkpoints trained with MLP aggregator
python train_lightning.py --aggregator_type mlp --resume path/to/checkpoint.pth

# For checkpoints trained with Transformer aggregator  
python train_lightning.py --aggregator_type transformer --resume path/to/checkpoint.pth
```

### Migration Strategy

If you have existing MLP checkpoints and want to use Transformer aggregator:

1. **Option A**: Continue training with MLP aggregator (`--aggregator_type mlp`)
2. **Option B**: Start fresh training with Transformer aggregator (`--aggregator_type transformer`)
3. **Option C**: Transfer backbone weights only (requires manual checkpoint surgery)

## Troubleshooting

### Memory Issues
If you encounter OOM errors with Transformer aggregator:
- Reduce `--agg_heads` to 1
- Use `--enable_gradient_checkpointing`
- Reduce `--batch_size`

### Performance Issues
If Transformer aggregator performs worse than MLP:
- Try `--agg_layers 2` for more capacity
- Increase `--agg_heads` to 4
- Verify your `--max_views` matches your data

### Debugging
Use the test script to verify aggregator functionality:
```bash
python -c "
import torch
from model.config import ModelConfig
from model.view_aggregator import ViewAggregator

config = ModelConfig(aggregator_type='transformer', max_views=8)
aggregator = ViewAggregator(config, 768)
features = torch.randn(2, 5, 768)
mask = torch.ones(2, 5, dtype=torch.bool)
output = aggregator.aggregate_views(features, mask)
print(f'Output shape: {output.shape}')  # Should be [2, 768]
"
```

## Performance Expectations

Based on our analysis of multi-view sports video understanding:

- **Transformer aggregator** should provide **better accuracy** for complex multi-view scenarios
- **Computational overhead** is minimal since it operates on already-extracted features
- **Memory overhead** is acceptable for typical view counts (4-8 views)

The Transformer's ability to model inter-view relationships should be particularly beneficial for foul detection where different camera angles provide complementary information about player contact, timing, and context. 