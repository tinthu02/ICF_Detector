# ICF Detector

ICF Detector is a machine-learning project for identifying likely slop or low-quality content from channel metadata and text signals.

## Project goal

The pipeline combines:
- feature engineering from channel and content metadata
- anomaly scoring
- model training and evaluation
- prediction utilities for new samples

## Repository structure

- `main.py` — entry point for running the project
- `src/` — data collection, feature extraction, model training, and prediction code
- `data/` — raw and processed datasets
- `notebooks/` — exploratory analysis and evaluation notebooks
- `models/` — trained model artifacts (ignored by Git unless you explicitly want to track them)

## Setup

1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

## Typical workflow

- Generate or refresh features:
  ```bash
  python src/features.py
  ```
- Train the model:
  ```bash
  python src/train.py
  ```
- Run predictions:
  ```bash
  python src/predict.py
  ```

## Notes

- The project expects the dataset files under `data/` to be available before training.
- Model outputs are not committed by default to keep the repository lightweight.

## Requirements

See [requirements.txt](requirements.txt) for the current dependency list.
