# Vendored: gridded-tnp (model code only)

Source: https://github.com/cambridge-mlg/gridded-tnp at commit `5b51367f412637ebf89da622270e8f75a280112c`
(Ashman, Diaconu, Langezaal, Weller, Turner, "Gridded Transformer Neural Processes for
Spatio-Temporal Data", ICML 2025). MIT licence, see `LICENSE`.

Copied: `tnp/models`, `tnp/networks`, `tnp/likelihoods`, and `tnp/utils/{batch,grids,helpers,distances}.py` and `tnp/utils/spherical_harmonics/`,
unchanged. Not copied: `tnp/data`, the experiment, Lightning and plotting utilities, `experiments/`.
Modified: `tnp/__init__.py` and `tnp/utils/__init__.py` import only the copied modules, so the
package needs nothing beyond torch, numpy, einops and check-shapes.

Installed in the project environment with `pip install --no-deps -e code/third_party/gridded_tnp`.
Used by `code/src/sense_energy/experiments/tnp_forecast.py`.
