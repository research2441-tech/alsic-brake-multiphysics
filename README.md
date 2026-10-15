# Al-2024 / nano-SiC brake disc: multiphysics and dynamic analysis

Companion code for "Multiphysics and Dynamic Finite-Element Analysis of Al-2024/Nano-SiC Composites Linking Measured Strengthening to Thermal Stress, Wear and Vibration of Automotive Brake Discs".

## Files
| File | Purpose |
|---|---|
| `fem.py` | In-house solvers: axisymmetric transient heat conduction, axisymmetric thermoelasticity, Hermite-element annular plate modal analysis |
| `verify_solvers.py` | Checks each solver against an analytical solution (Table IV) |
| `study.py` | Material data, micromechanics, strengthening efficiency, brake duty, wear, modal analysis, Monte Carlo |
| `make_results.py` | Monte Carlo statistics, sensitivities, desirability and Figs. 2–4 |
| `make_workflow_figure.py` | Fig. 1 |
| `nominal.json`, `mc_summary.json` | Stored outputs reported in the paper |

## Reproduce
```bash
pip install -r requirements.txt
python verify_solvers.py
python study.py nominal        # nominal results, efficiency table, mesh study, field files
python study.py mc 150 0       # 150-draw Monte Carlo (about 5 min on one core)
python make_results.py
python make_workflow_figure.py
```

## Notes
- Measured yield strength and Brinell hardness at 0–4 wt% SiC come from Niyaz Ahmed et al., Frattura ed Integrità Strutturale, 2025 (doi:10.3221/IGF-ESIS.72.11).
- The base Archard coefficient and its reduction ratio are derived from Ghahremani et al., Crystals, 2023 (doi:10.3390/cryst13050722).
- COMSOL Multiphysics equivalents of every model setting are listed in Table III of the paper.
