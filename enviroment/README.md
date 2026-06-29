# Environment setup (conda + Poetry)

PASTORS (Pastoral Activity & Soil-geochemistry Through Orbital Remote Sensing) uses a **hybrid conda + Poetry** environment:

- **conda** installs Python and the binary-heavy geospatial stack (GDAL, GEOS,
  PROJ, `rasterio`, `geopandas`, `pyproj`, `shapely`, `fiona`). These are
  painful to build from source via pip, so conda-forge handles them.
- **Poetry** manages the pure-Python application dependencies (numpy, pandas,
  scikit-learn, xgboost, optuna, tabpfn-client, composition-stats,
  earthengine-api, matplotlib, seaborn) and installs them **into the same
  conda env** (`poetry.toml` sets `virtualenvs.create = false`).

## One-time setup

```bash
# 1. Create and activate the conda env (Python + geospatial binaries)
conda env create -f enviroment/environment.yml
conda activate pastors

# 2. Install the Python dependencies with Poetry, into the conda env
cd enviroment
poetry install
cd ..
```

## Authentication (one-time, do NOT commit secrets)

```bash
# Google Earth Engine
earthengine authenticate

# TabPFN cloud client — set as an environment variable, never hard-code it
export TABPFN_TOKEN="your_token_here"
```

In code, read the token from the environment instead of pasting it:

```python
import os
from tabpfn_client import set_access_token
set_access_token(os.environ["TABPFN_TOKEN"])
```

## Updating dependencies

```bash
# geospatial / conda side
conda env update -f enviroment/environment.yml --prune

# python / poetry side
cd enviroment && poetry update && cd ..
```

## Notes

- Commit `poetry.lock` (generated on first `poetry install`) for reproducible
  builds.
- The geospatial packages are deliberately **absent** from `pyproject.toml` —
  they are owned by conda. Do not add them to Poetry, or you will get duplicate
  and possibly conflicting installs.
