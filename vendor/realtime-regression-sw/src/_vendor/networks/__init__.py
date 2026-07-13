# Vendored from regression-sw/src/networks/ @ 2d89767 on 2026-04-20 — DO NOT EDIT.
# Importing this package registers all 9 model factories with _registry:
#   - Time-series only: linear, transformer, tcn, patchtst, timesnet
#   - GNN variants: gnn (with gnn_temporal_type ∈ {transformer, tcn, bilstm, patch_transformer})
from . import linear  # noqa: F401 — registers "linear"
from . import transformer  # noqa: F401 — registers "transformer"
from . import tcn  # noqa: F401 — registers "tcn"
from . import patchtst  # noqa: F401 — registers "patchtst"
from . import timesnet  # noqa: F401 — registers "timesnet"
from . import gnn  # noqa: F401 — registers "gnn" (transformer/tcn/bilstm/patch_transformer)
from ._registry import create_model, list_models  # noqa: F401

__all__ = ["create_model", "list_models"]
