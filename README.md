# Random Forest Key Classifier (Major/Minor)

This project predicts whether an audio file is in **major** or **minor** mode using a trained RandomForest model.

## Requirements
- macOS
- Python 3.11 (recommended)

## Dataset Notes

The raw audio dataset (WAV files) is **not included** in this repository due to size constraints.

This project uses **precomputed audio features** stored in:
audio_features/audio_features_with_labels_plus_tonnetz_cqt.csv

Because of this:
- `train.py` can run **without raw audio files**
- Validation accuracy and confusion matrix are reproducible
- The final trained model is provided as `rf_key_classifier.joblib`

To retrain from raw audio, the full dataset must be downloaded separately and placed in the expected `fmak_wav/` directory structure.

## Setup
From the project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
## Running Predictions

This project includes a `predict.py` script for running inference using the
pretrained Random Forest model (`rf_key_classifier.joblib`).

You must provide either:
- a single audio file using `--file`, or
- a directory of audio files using `--dir`.

### Predict a single audio file

```bash
python predict.py --file path/to/audio.wav

# Example
python predict.py --file wavs-sample/example.wav
```

### Predict all files in a directory
python predict.py --dir path/to/folder

``` bash
# Example
python predict.py --dir wavs-sample
```

``` bash
# (Optional): Specifiy a custome model path
python predict.py --file wavs-sample/example.wav --model rf_key_classifier.joblib

```
### Output
For each audio file, the script prints the predicted mode:
example.wav → major

or

example.wav → minor

Dataset Notes:
The raw audio dataset (WAV files) is not included in this repository due to size constraints.

The predict.py script works using the provided trained model without
requiring the original audio dataset.

This project uses precomputed audio features stored in:
audio_features/audio_features_with_labels_plus_tonnetz_cqt.csv

Because of this:

train.py can run without raw audio files

Validation accuracy and confusion matrix are reproducible

The final trained model is provided as rf_key_classifier.joblib

To retrain from raw audio, the full dataset must be downloaded separately and placed in the expected fmak_wav/ directory structure.
