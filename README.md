# ECG Beat Classifier — Streamlit App

A minimal Streamlit front end for the grayscale baseline CNN trained in
`heart_disease_prevention_using_artificial_intelligence.py`. Upload an ECG
image and it returns the predicted beat class and confidence.

## 1. Before you run anything

Your model file is 155MB, which is over GitHub's 100MB push limit, so it
does **not** go in the repo. Instead `app.py` downloads it once at startup
from your Google Drive file (`GDRIVE_FILE_ID` is already set in `app.py`),
then caches it on disk.

### Make the Drive file downloadable

The app downloads server-side with no Google login, so the file has to be
public:

1. In Google Drive, right-click `baseline_cnn.keras` → **Share**.
2. Under "General access", change it from "Restricted" to
   **"Anyone with the link"**, role **Viewer**.
3. Save. If you skip this, the download pulls down an HTML "request access"
   page instead of the model, and the app will show an error telling you to
   fix sharing.

`app.py` uses `gdown`, which handles Drive's confirmation step for files
this large (plain HTTP requests get blocked by Drive's "can't scan for
viruses" interstitial at this size).

**One caveat:** Google enforces a daily download quota on publicly shared
files. It's generous for normal use, but if the app's container restarts
often (Streamlit Community Cloud does this periodically) or the app gets
heavy traffic, you could occasionally see a "too many users have viewed or
downloaded this file" error for ~24 hours. If that happens in practice,
moving the file to a free Hugging Face model repo (no quota, same
download-on-startup pattern) is the fix — ask if you want that swapped in.

### Check the class order

Open `app.py` and check `CLASS_NAMES`. It currently assumes the
alphabetical order Keras generates automatically: `["F", "M", "N", "Q", "S", "V"]`.
Compare this against the `Class names:` line your training notebook printed
(`train_ds.class_names`) — if the order differs, update the list, or the
predicted label will be wrong even though the math is right.

Final folder layout (no model file in the repo):

```
your-repo/
├── app.py
├── requirements.txt
└── README.md
```

## 2. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at `http://localhost:8501`. On first run it downloads the
model from Google Drive; after that it reuses the local copy.

## 3. Deploy to Streamlit Community Cloud (free)

1. Push the three files above to a GitHub repo (public or private) — the
   model file stays out of git entirely now.
2. Go to share.streamlit.io and sign in with GitHub.
3. Click "New app", pick the repo/branch, and set the main file path to
   `app.py`.
4. Deploy. The first build installs TensorFlow, and the first page load
   downloads the model — both take a few minutes, after that it's cached.

**Memory:** Streamlit Community Cloud's free tier gives ~1 GB RAM. The
grayscale baseline CNN is small and should run comfortably within that —
this is part of why it's a better fit for free hosting than the DenseNet121
version.

## 4. Notes

- The app mirrors the exact preprocessing used in training
  (`rgb_to_grayscale` → normalize to `[0, 1]` → resize to 224×224), so
  predictions should match what you saw in the notebook's evaluation.
- The in-app disclaimer is there because this is a classroom/demo project,
  not a diagnostic tool — keep it, or adapt the wording, but don't present
  predictions as medical advice.
