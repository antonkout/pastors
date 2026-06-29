# PASTORS

**P**astoral **A**ctivity & **S**oil-geochemistry **T**hrough **O**rbital **R**emote **S**ensing

Detecting anthropogenic activity in pastoral landscapes by fusing high-resolution
optical satellite imagery (SPOT) with multispectral Sentinel-2 and linking it to
ground soil geochemistry analysed under the Compositional Data Analysis (CoDA)
framework. The prediction target is a **symmetric compositional balance**
contrasting activity-marker elements against the natural geochemical background;
markers and background are selected automatically from a **principal-balance
(PBA) dendrogram**, and a bare-soil mask restricts inference to exposed soil. Two
regressors are benchmarked: **XGBoost** and the tabular foundation model **TabPFN**.

Protocol developed in Khor Rori (Dhofar, Oman) and tested for transferability in
Turkana, Kenya.

## Repository layout

```
pastors/
  helpers/        pipeline modules (balances, dendrogram, fusion, masking, models)
  utils/          lookup tables (LOD, oxide factors); TabPFN token (git-ignored)
  jn/             notebooks (PBA dendrogram, TabPFN/Colab)
  enviroment/     conda + Poetry environment definition
  pastors.ipynb   main XGBoost pipeline
  baresoil_mask.ipynb   LULC-derived vegetation / bare-soil mask
  data/           input data (git-ignored)
  output/         results / rasters (git-ignored)
```

> `data/` and `output/` are intentionally excluded from version control. The
> TabPFN API token lives in `utils/tabpfn_token.json`, which is **git-ignored** —
> create it locally as `{"token": "<your_token>"}`.

## Setup

See [`enviroment/README.md`](enviroment/README.md) for the conda + Poetry setup.

```bash
conda env create -f enviroment/environment.yml
conda activate pastors
cd enviroment && poetry install && cd ..
```

## Method

1. **Compositional pre-processing** — zero replacement, closure, log-ratio coordinates.
2. **Automated marker/background selection** — cluster parts by Aitchison
   (log-ratio) variation; read a sequential binary partition; rank principal balances.
3. **Population split** — GMM + BIC on the rank-1 balance; antimode = enrichment threshold.
4. **Satellite fusion & features** — SPOT + Sentinel-2 + spectral indices (NDVI, BSI, SAVI, MSAVI, ...).
5. **Vegetation / bare-soil masking** — from LULC; inference restricted to bare soil.
6. **Modelling** — XGBoost (tuned) vs. TabPFN; evaluated on internal hold-out and external validation.

## License

MIT (see `LICENSE`).
