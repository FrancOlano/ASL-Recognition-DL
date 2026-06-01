# ASL Recognition Demo

This folder contains a small browser demo for the ASL fingerspelling model.

## What it does

- Opens the user camera in the browser.
- Sends live frames to a Python backend for inference.
- Recognizes one static ASL letter at a time.
- Appends stable predictions into a text area so the user can spell words.

## Checkpoint loading

The app loads the fixed checkpoint at `models/checkpoints/best_mobilenet_v2_scratch.pth`.

If no checkpoint is present, the page still loads but the prediction endpoint stays disabled until a checkpoint is added.

## Run it

1. Install the demo dependency:

   ```bash
   pip install Flask
   ```

2. Start the app from the repository root:

   ```bash
   python demo/app.py
   ```

3. Open `http://127.0.0.1:5000` in a browser and allow camera access.

## Notes

- The model uses the same `200x200` RGB preprocessing as the training config in `engine/`.
- The class order is alphabetical over the 26 letters (A-Z) and is read from `results/classes.json` when available.