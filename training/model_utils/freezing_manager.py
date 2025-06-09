class AdvancedFreezingManager:
    """
    Enhanced freezing manager with comprehensive multi-metric analysis.
    Uses gradient magnitude, stability, performance trends, and layer dependencies.
    """
    
    def __init__(self, model, importance_threshold=0.002, performance_threshold=0.001, 
                 analysis_window=2, enable_rollback=True, enable_dependency_analysis=True,
                 gradient_momentum=0.9, device='cuda'):
        """
        Initialize advanced freezing manager with enhanced capabilities.
        
        Args:
            model: PyTorch model
            importance_threshold: Lower threshold for considering layer unfreeze (reduced from 0.005)
            performance_threshold: Minimum improvement needed (reduced from 0.002)
            analysis_window: Epochs to look back for trend analysis (reduced from 3)
            enable_rollback: Whether to rollback on performance degradation
            enable_dependency_analysis: Whether to analyze layer dependencies
            gradient_momentum: Momentum for gradient tracking
            device: Device for computations
        """
        self.model = model
        self.importance_threshold = importance_threshold
        self.performance_threshold = performance_threshold
        self.analysis_window = analysis_window
        self.enable_rollback = enable_rollback
        self.enable_dependency_analysis = enable_dependency_analysis
        self.gradient_momentum = gradient_momentum
        self.device = device
        
        # Performance tracking
        self.performance_history = []
        self.unfrozen_layers = set()
        self.warmup_layers = {}  # layer_name -> epochs_since_unfreeze
        self.frozen_layers = set()
        
        # Gradient tracking with enhanced metrics
        self.gradient_history = {}
        self.gradient_magnitude_ema = {}
        self.gradient_stability_score = {}
        self.layer_importance_score = {}
        
        # Rollback mechanism
        self.rollback_buffer = {}
        self.performance_before_unfreeze = {}
        
        # Layer dependency graph
        self.layer_dependencies = {}
        
        # Initialize tracking
        self._setup_gradient_tracking()
        self._analyze_layer_dependencies()
        
        logger.info(f"[ADVANCED_FREEZING] Initialized with enhanced multi-metric analysis")
        logger.info(f"  - Base importance threshold: {self.importance_threshold:.6f}")
        logger.info(f"  - Performance threshold: {self.performance_threshold:.6f}")
        logger.info(f"  - Analysis window: {self.analysis_window} epochs")
        logger.info(f"  - Rollback enabled: {self.enable_rollback}")
        logger.info(f"  - Dependency analysis: {self.enable_dependency_analysis}")
        
        # Start with backbone frozen
        self.freeze_backbone()
    
    def _calculate_adaptive_threshold(self, epoch):
        """Calculate adaptive importance threshold based on training progress."""
        # Start more aggressive, become more conservative
        progress_factor = min(epoch / 10.0, 1.0)  # Normalize over first 10 epochs
        adaptive_threshold = self.importance_threshold * (0.5 + 0.5 * progress_factor)
        return adaptive_threshold 

    def emergency_unfreeze(self, min_layers=2):
        """
        Emergency unfreezing when normal criteria aren't met.
        Forces unfreezing of the most promising layers.
        """
        # Get all frozen backbone layers
        frozen_backbone_layers = []
        for name, module in self.model.named_modules():
            if self._is_backbone_layer(name) and name in self.frozen_layers:
                frozen_backbone_layers.append(name)
        
        # Sort by importance score (if available) or use predefined order
        if self.layer_importance_score:
            frozen_backbone_layers.sort(
                key=lambda x: self.layer_importance_score.get(x, 0), 
                reverse=True
            )
        else:
            # Default order: later layers first (closer to head)
            layer_priorities = ['layer4', 'layer3', 'layer2', 'layer1']
            frozen_backbone_layers.sort(
                key=lambda x: next((i for i, p in enumerate(layer_priorities) if p in x), 999)
            )
        
        # Unfreeze top candidates
        unfrozen_count = 0
        for layer_name in frozen_backbone_layers[:min_layers]:
            if self._unfreeze_layer(layer_name):
                self.unfrozen_layers.add(layer_name)
                self.frozen_layers.discard(layer_name)
                unfrozen_count += 1
                logger.info(f"[EMERGENCY_UNFREEZE] Unfroze layer: {layer_name}")
        
        if unfrozen_count > 0:
            logger.info(f"[EMERGENCY_UNFREEZE] Successfully unfroze {unfrozen_count} layers")
            return True
        
        return False 