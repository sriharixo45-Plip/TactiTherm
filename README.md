# TactiTherm

**Low-Cost Controlled-Force Tactile–Thermal Mapping for Localized Soft-Tissue Abnormality Detection**

TactiTherm is an experimental prototype that combines an Arduino/Wokwi simulation, a Python processing bridge, engineered features, and a Random Forest classifier to demonstrate four-class spatial mapping from simulated tactile-related and thermal inputs.

> **Research prototype notice:** The current inputs are simulated using potentiometers and the datasets are synthetic. This project is not a clinically validated diagnostic device and must not be used for medical diagnosis.

## Team

- Sri Hari Kumaran A — CB.SC.U4CSE25656
- Lohit Kumar R E — CB.SC.U4CSE25636
- Eswara Sudhan H — CB.SC.U4CSE25619
- Sabari D — CB.SC.U4CSE25650

Faculty Guide: Krishna Priya G.
Institution: Amrita Vishwa Vidyapeetham, Coimbatore

## Repository contents

- `tactile_thermal_project.ino` — Arduino firmware
- `diagram.json` — Wokwi circuit diagram
- `wokwi.toml` — Wokwi configuration
- `ml_bridge.py` — Python serial bridge and inference workflow
- `tactile_thermal_model_v3.pkl` — trained seven-feature Random Forest artifact used by the bridge
- `dataset1_clean.csv`, `dataset2_clean.csv` — synthetic datasets
- `train_model.py` — trains the seven-feature and two-input baseline models
- `evaluate_model.py`, `analyze_dataset.py`, `diagnose_features.py`, `inspect_model.py` — analysis and evaluation utilities
- `build/` — compiled firmware artifacts referenced by `wokwi.toml`

## Requirements

- Python 3.10 or later (use a Python version supported by the installed packages)
- Wokwi simulator / Wokwi VS Code extension for the Arduino simulation workflow
- Python packages listed in `requirements.txt`

Install Python dependencies from the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Running the Python bridge

1. Start the Wokwi simulation with the RFC2217 serial server enabled on port `4000`.
2. In a terminal opened in this repository directory, install the requirements.
3. Run:

```bash
python ml_bridge.py
```

The bridge is configured for `rfc2217://localhost:4000` at 115200 baud and loads `tactile_thermal_model_v3.pkl` from the same directory. Adjust the port only if your simulation setup differs.

## Training and evaluation

To train the models from the included CSV datasets, run:

```bash
python train_model.py
```

The training script uses a group-based train/test split and writes model artifacts to the project directory. Training may take time depending on your machine.

## Reported experimental results

The recorded seven-feature model achieved 79.10% accuracy and a macro F1-score of 0.5956 on the synthetic test split. The two-input force/temperature baseline achieved 76.25% accuracy and a macro F1-score of 0.5442. These are experimental results on synthetic data, not clinical performance measures.

## Model artifact note

The trained model is a binary Python pickle/joblib artifact. Only load it from a trusted source. Re-training can be performed with `train_model.py` using the supplied datasets.

## License

No license is included in this package. Add a license only after confirming that the team has the right to license all included code and assets.
