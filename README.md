# Thoroughbred Speed Figure & Race Analyzer

Streamlit app for creating proprietary speed figures from Brisnet-style `.drf` / `.dr2` / `.dr3` / `.dr4` race cards and identifying the most likely winner in each race.

## Features
- Upload one or more Brisnet card files (`.drf`, `.dr2`, `.dr3`, `.dr4`) for a race card.
- Analyzer tabs for every track included in the uploaded card (with US thoroughbred track mapping).
- Optional upload of historical track results (`.csv` / `.xlsx`) to calculate track-bias adjustments.
- Integrates detected track biases into current-race proprietary speed figures.
- Pace predictor graphic per race showing early/middle/late pace projections per horse.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected input columns
The Brisnet card parser is tolerant and tries to map common columns to:
- `track`, `race`, `horse`, `post`, `surface`, `distance`
- `early_pace`, `middle_pace`, `late_pace`
- `last_speed`, `avg_speed`, `class_rating`, `days_since`, `run_style`

Historical file preferred columns:
- `track`, `post`, `finish_position`, `run_style`


> Note: The uploader accepts any file selection and validates extensions in-app to avoid browser/Streamlit extension-filter quirks.
