# RTE Benchmarks

Radiative Transfer Equation (RTE) benchmarks using [ARTS](https://www.radiativetransfer.org/) (Atmospheric Radiative Transfer Simulator). The benchmarks compute broadband shortwave (SW) and longwave (LW) radiative fluxes for several standard atmospheric test cases and compare them against reference flux data.

## Test Cases

| Setup     | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| `ckdmip`  | CKDMIP (Correlated K-Distribution Model Intercomparison Project) atmospheres |
| `rce`     | Radiative-Convective Equilibrium atmospheres                               |
| `rfmip`   | RFMIP (Radiative Forcing Model Intercomparison Project) atmospheres       |

## Repository Structure

```
data/            # Input atmospheric profiles and auxiliary data (ARTS XML)
results/         # Reference and computed flux outputs (NetCDF)
  <setup>/LW/
  <setup>/SW/
scripts/         # Python scripts and helper modules
  cache/         # LUT caches per setup / spectral band
```

## Scripts

| File                            | Purpose                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| `convert_rte-examples2arts2.py` | Convert rte-examples netCDF data to ARTS `ArrayOfGriddedField4` XML      |
| `rte_benchmarks.py`             | Run single-profile SW/LW benchmark simulations (slower, more flexible to adjust outputs) |
| `rte_benchmarks_batch.py`       | Run batch SW/LW benchmark simulations (faster, less flexible)     |
| `rte_aux_functions.py`          | Auxiliary functions (thermodynamics, unit conversions, flux computations) |

## Dependencies

- [pyarts](https://github.com/atmtools/arts) — ARTS Python interface
- [pyarts-fluxes >= 0.6](https://github.com/atmtools/pyarts-fluxes) — flux computation
- numpy, scipy, xarray

## Workflow

### 1. Convert Input Data

Convert the rte-examples NetCDF data to ARTS XML format:

```bash
cd scripts
python convert_rte-examples2arts2.py
```

This reads source files from `data/` and writes ARTS-compatible XML files.

### 2. Run Benchmarks

**Single-profile mode**:

```bash
python rte_benchmarks.py
```

**Batch mode**:

```bash
python rte_benchmarks_batch.py
```

Results are written to `results/<setup>/LW/` and `results/<setup>/SW/` as NetCDF files.

## Author

Manfred Brath
