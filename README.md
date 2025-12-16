# Random Forest Key Classifier (Major/Minor)

This project predicts whether an audio file is in **major** or **minor** mode using a trained RandomForest model.

## Requirements
- macOS
- Python 3.11 (recommended)

## Setup

## Dataset Notes

The raw audio dataset (WAV files) is **not included** in this repository due to size constraints.

This project uses **precomputed audio features** stored in:
audio_features/audio_features_with_labels_plus_tonnetz_cqt.csv

Because of this:
- `train.py` can run **without raw audio files**
- Validation accuracy and confusion matrix are reproducible
- The final trained model is provided as `rf_key_classifier.joblib`

To retrain from raw audio, the full dataset must be downloaded separately and placed in the expected `fmak_wav/` directory structure.



From the project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

