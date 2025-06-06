#!/usr/bin/env python3
"""
Simplified training script using the modular training package.

This is the new entry point that replaces the original monolithic train.py file.
All functionality has been moved to the training/ package for better organization.
"""

from training.train import main

if __name__ == "__main__":
    main() 