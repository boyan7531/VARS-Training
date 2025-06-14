"""
Data loading optimization utilities for improved training performance.

This module provides:
- Adaptive prefetching pipeline optimization
- Data loading bottleneck detection
- Intelligent batch sizing based on GPU memory
"""

import torch
import time
import psutil
import logging
from collections import deque
from threading import Thread, Event
import queue

logger = logging.getLogger(__name__)


class DataLoadingProfiler:
    """Profiles data loading vs compute time to detect bottlenecks."""
    
    def __init__(self, window_size=20, bottleneck_threshold=0.3):
        self.window_size = window_size
        self.bottleneck_threshold = bottleneck_threshold
        self.data_times = deque(maxlen=window_size)
        self.compute_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        self.batch_count = 0
        self.warnings_issued = 0
        self.max_warnings = 3  # Limit warning spam
        
    def record_batch_timing(self, data_time, compute_time):
        """Record timing for a single batch."""
        total_time = data_time + compute_time
        self.data_times.append(data_time)
        self.compute_times.append(compute_time)
        self.total_times.append(total_time)
        self.batch_count += 1
        
        # Analyze every 10 batches after initial window
        if self.batch_count >= self.window_size and self.batch_count % 10 == 0:
            self._analyze_bottleneck()
    
    def _analyze_bottleneck(self):
        """Analyze if data loading is a bottleneck."""
        if len(self.data_times) < self.window_size:
            return
            
        avg_data_time = sum(self.data_times) / len(self.data_times)
        avg_compute_time = sum(self.compute_times) / len(self.compute_times)
        avg_total_time = sum(self.total_times) / len(self.total_times)
        
        data_ratio = avg_data_time / avg_total_time if avg_total_time > 0 else 0
        
        # If data loading takes more than threshold% of total time, it's a bottleneck
        if data_ratio > self.bottleneck_threshold and self.warnings_issued < self.max_warnings:
            self.warnings_issued += 1
            logger.warning(f"🚨 DATA LOADING BOTTLENECK DETECTED!")
            logger.warning(f"   Data loading: {avg_data_time:.3f}s ({data_ratio*100:.1f}% of total time)")
            logger.warning(f"   Compute time: {avg_compute_time:.3f}s")
            logger.warning(f"   Recommendations:")
            logger.warning(f"   - Increase --num_workers (currently using {torch.utils.data.get_worker_info() or 'unknown'})")
            logger.warning(f"   - Increase --prefetch_factor")
            logger.warning(f"   - Enable --pin_memory if not already enabled")
            logger.warning(f"   - Consider reducing data augmentation complexity")
    
    def get_stats(self):
        """Get current performance statistics."""
        if not self.data_times:
            return {}
            
        return {
            'avg_data_time': sum(self.data_times) / len(self.data_times),
            'avg_compute_time': sum(self.compute_times) / len(self.compute_times),
            'data_bottleneck_ratio': sum(self.data_times) / sum(self.total_times) if sum(self.total_times) > 0 else 0,
            'batches_analyzed': len(self.data_times)
        }


class AdaptivePrefetcher:
    """Adaptive prefetching system that adjusts based on performance."""
    
    def __init__(self, initial_prefetch_factor=2, max_prefetch_factor=8):
        self.current_prefetch_factor = initial_prefetch_factor
        self.max_prefetch_factor = max_prefetch_factor
        self.performance_history = deque(maxlen=10)
        self.adjustment_cooldown = 50  # Batches between adjustments
        self.last_adjustment = 0
        
    def should_adjust_prefetch(self, batch_idx, data_ratio):
        """Determine if prefetch factor should be adjusted."""
        if batch_idx - self.last_adjustment < self.adjustment_cooldown:
            return False, self.current_prefetch_factor
            
        self.performance_history.append(data_ratio)
        
        if len(self.performance_history) < 5:
            return False, self.current_prefetch_factor
            
        avg_data_ratio = sum(self.performance_history) / len(self.performance_history)
        
        # If data loading is consistently slow, increase prefetch
        if avg_data_ratio > 0.25 and self.current_prefetch_factor < self.max_prefetch_factor:
            self.current_prefetch_factor = min(self.current_prefetch_factor + 1, self.max_prefetch_factor)
            self.last_adjustment = batch_idx
            logger.info(f"🚀 Increased prefetch factor to {self.current_prefetch_factor} to reduce data loading bottleneck")
            return True, self.current_prefetch_factor
            
        # If data loading is very fast, decrease prefetch to save memory
        elif avg_data_ratio < 0.1 and self.current_prefetch_factor > 1:
            self.current_prefetch_factor = max(self.current_prefetch_factor - 1, 1)
            self.last_adjustment = batch_idx
            logger.info(f"💾 Decreased prefetch factor to {self.current_prefetch_factor} to save memory")
            return True, self.current_prefetch_factor
            
        return False, self.current_prefetch_factor


class IntelligentBatchSizer:
    """Dynamically adjusts batch size based on GPU memory usage."""
    
    def __init__(self, initial_batch_size, min_batch_size=1, max_batch_size=64, memory_target=0.85):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_target = memory_target  # Target 85% GPU memory usage
        self.adjustment_history = []
        self.stable_epochs = 0
        self.adjustment_cooldown = 3  # Epochs between adjustments
        
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage ratio."""
        if not torch.cuda.is_available():
            return 0.0
            
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return allocated / total, cached / total
    
    def should_adjust_batch_size(self, epoch, oom_occurred=False):
        """Determine if batch size should be adjusted."""
        allocated_ratio, cached_ratio = self.get_gpu_memory_usage()
        
        # Immediate reduction on OOM
        if oom_occurred:
            new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
            if new_batch_size != self.current_batch_size:
                logger.warning(f"🚨 OOM detected! Reducing batch size: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                self.stable_epochs = 0
                return True, self.current_batch_size
            return False, self.current_batch_size
        
        # Only adjust every few epochs for stability
        if self.stable_epochs < self.adjustment_cooldown:
            self.stable_epochs += 1
            return False, self.current_batch_size
        
        # Conservative memory management
        if allocated_ratio > 0.9:  # Very high memory usage
            new_batch_size = max(self.current_batch_size - 1, self.min_batch_size)
            if new_batch_size != self.current_batch_size:
                logger.info(f"💾 High memory usage ({allocated_ratio*100:.1f}%). Reducing batch size: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                self.stable_epochs = 0
                return True, self.current_batch_size
                
        elif allocated_ratio < self.memory_target * 0.7:  # Low memory usage, could increase
            new_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
            if new_batch_size != self.current_batch_size:
                logger.info(f"📈 Low memory usage ({allocated_ratio*100:.1f}%). Increasing batch size: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                self.stable_epochs = 0
                return True, self.current_batch_size
        
        return False, self.current_batch_size
    
    def get_effective_batch_size_multiplier(self):
        """Get multiplier to maintain effective batch size when using gradient accumulation."""
        return max(1, self.initial_batch_size // self.current_batch_size)


class OptimizedDataLoader:
    """Enhanced DataLoader with performance monitoring and adaptive optimizations."""
    
    def __init__(self, dataset, batch_size, num_workers=4, prefetch_factor=2, 
                 pin_memory=True, persistent_workers=True, enable_profiling=False,
                 enable_adaptive_prefetch=True, **kwargs):
        
        self.dataset = dataset
        self.initial_batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.kwargs = kwargs
        
        # Performance monitoring
        self.profiler = DataLoadingProfiler() if enable_profiling else None
        self.adaptive_prefetcher = AdaptivePrefetcher(prefetch_factor) if enable_adaptive_prefetch else None
        
        # Create initial dataloader
        self.dataloader = self._create_dataloader(batch_size, prefetch_factor)
        self.batch_count = 0
        
    def _create_dataloader(self, batch_size, prefetch_factor):
        """Create a DataLoader with specified parameters."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            prefetch_factor=prefetch_factor if self.num_workers > 0 else None,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            **self.kwargs
        )
    
    def __len__(self):
        """Return the length of the underlying dataloader."""
        return len(self.dataloader)
    
    def __iter__(self):
        """Iterate through the dataloader with performance monitoring."""
        data_iter = iter(self.dataloader)
        
        for batch in data_iter:
            start_time = time.time()
            
            # Yield the batch and measure data loading time
            yield batch
            
            data_time = time.time() - start_time
            self.batch_count += 1
            
            # Record timing for profiling (compute time will be recorded externally)
            if self.profiler:
                # For now, just record data time. Compute time needs to be provided externally
                pass
    
    def record_compute_time(self, compute_time):
        """Record compute time for the last batch."""
        if self.profiler and hasattr(self, '_last_data_time'):
            self.profiler.record_batch_timing(self._last_data_time, compute_time)
    
    def update_prefetch_if_needed(self):
        """Update prefetch factor if adaptive prefetching suggests it."""
        if not self.adaptive_prefetcher:
            return False
            
        stats = self.profiler.get_stats() if self.profiler else {}
        data_ratio = stats.get('data_bottleneck_ratio', 0)
        
        should_adjust, new_prefetch = self.adaptive_prefetcher.should_adjust_prefetch(
            self.batch_count, data_ratio
        )
        
        if should_adjust and new_prefetch != self.prefetch_factor:
            self.prefetch_factor = new_prefetch
            # Recreate dataloader with new prefetch factor
            self.dataloader = self._create_dataloader(self.dataloader.batch_size, self.prefetch_factor)
            return True
            
        return False
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        stats = {'batch_count': self.batch_count}
        if self.profiler:
            stats.update(self.profiler.get_stats())
        return stats


def create_optimized_dataloader(dataset, batch_size, num_workers=4, prefetch_factor=2,
                               pin_memory=True, persistent_workers=True, enable_optimizations=True,
                               **kwargs):
    """Create an optimized DataLoader with performance monitoring."""
    
    if enable_optimizations:
        return OptimizedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs
        )
    else:
        # Fallback to standard DataLoader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            **kwargs
        ) 