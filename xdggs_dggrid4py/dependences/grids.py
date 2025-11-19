from xdggs.grid import DGGSInfo
from typing import Any, ClassVar, Self
from dataclasses import dataclass
import numpy as np
import os
import decimal
import tempfile
import shapely

from dggrid4py import DGGRIDv8
from xdggs_dggrid4py.utils import _geodetic_to_authalic, _authalic_to_geodetic


try:
    dggrid_path = os.environ['DGGRID_PATH']
except KeyError:
    raise Exception("DGGRID_PATH env var not found")

zone_id_repr_list = {"int": ["INT64", np.nan], "hexstring": ["INT64", ''], "textural": ["DIGIT_STRING", '']}

GridsConfig = {'IGEO7': {"refinement_level_range": range(0, 22),
                         "meta_config": {"input_address_type":  'HIERNDX',
                                        "input_hier_ndx_system": 'Z7',
                                        "input_hier_ndx_form":  'INT64',  # defaults to int representation
                                        "output_address_type":  'HIERNDX',
                                        "output_cell_label_type":  'OUTPUT_ADDRESS_TYPE',
                                        "output_hier_ndx_system":  'Z7',
                                        "output_hier_ndx_form":  'INT64',  # defaults to int representation
                                        "dggs_vert0_lon": 11.20
                                        }
                         },
               'ISEA7H': {"refinement_level_range": range(0, 22),
                          "meta_config": {"input_address_type":  'HIERNDX',
                                         "input_hier_ndx_system": 'SEQNUM',
                                         "input_hier_ndx_form":  'INT64',  # defaults to int representation
                                         "output_address_type":  'HIERNDX',
                                         "output_cell_label_type":  'OUTPUT_ADDRESS_TYPE',
                                         "output_hier_ndx_system":  'SEQNUM',
                                         "output_hier_ndx_form":  'INT64',  # defaults to int representation
                                         "dggs_vert0_lon": 11.20
                                         }
                          },
               }


@dataclass(frozen=True)
class IGEO7Info(DGGSInfo):
    grid_name: str
    _dggrid_meta_config: dict
    _wgs84_geodetic_conversion: bool = True
    _dggs_vert0_lon: decimal.Decimal | float = 11.20
    _dggrid = DGGRIDv8(dggrid_path, tempfile.TemporaryDirectory().name, silent=True)

    valid_parameters: ClassVar[dict[str, Any]] = {"level": GridsConfig["IGEO7"]["refinement_level_range"]}

    def __post_init__(self):
        if (self.grid_name.upper() != 'IGEO7'):
            raise ValueError('Wrong grid_name for IGEO7Info.')
        if self.level not in self.valid_parameters["level"]:
            raise ValueError(f'level must be an integer between {GridsConfig["IGEO7"]["refinement_level_range"]}.')

    @classmethod
    def from_dict(cls: type[Self], mapping: dict[str, Any]) -> Self:
        params = {k: v for k, v in mapping.items()}
        return cls(**params)

    def to_dict(self: Self) -> dict[str, Any]:
        return {"level": self.level, "grid_name": self.grid_name,
                "_wgs84_geodetic_conversion": self._wgs84_geodetic_conversion,
                "_dggs_vert0_lon": self._dggs_vert0_lon}

    def cell_ids2geographic(
        self, cell_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        centroids_df = self._dggrid.grid_cell_centroids_from_cellids(cell_ids, self.grid_name, self.level,
                                                                     **self._dggrid_meta_config)
        centroids_df.geometry = _authalic_to_geodetic(centroids_df.geometry, self._wgs84_geodetic_conversion)
        centroids = centroids_df.get_coordinates()
        return (centroids['x'].values, centroids['y'].values)

    def geographic2cell_ids(self, lon, lat):
        assert len(lon) == len(lat), f"{__name__} the length of lon and lat are not equal"
        centroids = GeoSeries([shapely.Point(c[0], c[1]) for c in zip(lon, lat)])
        centroids = _geodetic_to_authalic(centroids, self._wgs84_geodetic_conversion)
        centroids = self._dggrid.cells_for_geo_points(centroids, True, self.grid_name, self.level, **self._dggrid_meta_config)
        return centroids['name'].values

    def cell_boundaries(self, cell_ids, backend="shapely"):
        if (backend != "shapely"):
            raise NotImplementedError("Only shapely is implemeneted")
        hexagon_df = self._dggrid.grid_cell_polygons_from_cellids(cell_ids, self.grid_name,
                                                                  self.level,
                                                                  **self._dggrid_meta_config)
        geometry = _authalic_to_geodetic(hexagon_df.geometry, self._wgs84_geodetic_conversion)
        return geometry.values

    def zoom_to(self, cell_ids, level: int):
        raise NotImplementedError("")
