from xdggs_dggrid4py.regridding.regridding import mapblocks_regridding, regridding
# Eagerly import the method modules so their @register_regridding_method
# decorators populate the registry. Without this, callers of
# mapblocks_regridding(method='mapblocks_nearestcentroid') get a confusing
# "method not found" error unless they happen to have imported the method
# module elsewhere.
from xdggs_dggrid4py.regridding.methods.centroid_based import (  # noqa: F401
    centerpoint, nearestcentroid, mapblocks_nearestcentroid,
)
__all__ = ['mapblocks_regridding', 'regridding']
