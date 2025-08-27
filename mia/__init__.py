"""
Membership Inference Attack (MIA) package
"""

from .lira import LiRAAttacker, create_shadow_datasets

__all__ = ['LiRAAttacker', 'create_shadow_datasets']