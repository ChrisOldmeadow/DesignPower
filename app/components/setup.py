"""
Setup module for components to handle path configuration.

This module adds the project root to the Python path so components can import from core.
"""
import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
