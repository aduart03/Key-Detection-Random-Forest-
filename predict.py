import argparse
import os
from glob import glob

import joblib
import numpy as np
import pandas as pd
import librosa


def extract_features(file_path: str, sr: int = 22050) -> dict:
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = {
        "chroma_stft_mean": float(np.mean(chroma_stft)),
        "spectral_centroid_mean": float(np.mean(spec_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spec_bw)),
        "rolloff_mean": float(np.mean(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
    }

    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))

    return feats


def predict_one(model, file_path: str) -> str:
    feats = extract_features(file_path)
    X = pd.DataFrame([feats])

    # Ensure column order matches training (important!)
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0.0)

    pred = model.predict(X)[0]
    return str(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, help="Path to a single audio file")
    ap.add_argument("--dir", type=str, help="Directory of audio files (wav/mp3/flac)")
    ap.add_argument("--model", type=str, default="rf_key_classifier.joblib", help="Path to saved .joblib model")
    args = ap.parse_args()

    if not args.file and not args.dir:
        ap.error("Provide --file or --dir")

    model = joblib.load(args.model)

    if args.file:
        print(os.path.basename(args.file), "->", predict_one(model, args.file))
        return

    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(args.dir, ext)))

    if not files:
        raise SystemExit(f"No audio files found in {args.dir}")

    for f in sorted(files):
        try:
            print(os.path.basename(f), "->", predict_one(model, f))
        except Exception as e:
            print(os.path.basename(f), "-> ERROR:", e)


if __name__ == "__main__":
    main()
