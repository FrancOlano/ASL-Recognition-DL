# ASL Recognition Demo

This folder contains a small browser demo for the ASL fingerspelling model.

## What it does

- Opens the user camera in the browser.
- Sends live frames to a Python backend for inference.
- Recognizes one static ASL letter at a time.
- Appends stable predictions into a text area so the user can spell words.

## Checkpoint loading

The app looks for the newest checkpoint in `models/checkpoints/` with one of these extensions:

- `.pth`
- `.pt`
- `.ckpt`

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

- The model uses the same `224x224` RGB preprocessing as the training code in `src/`.
- The class order is alphabetical over the 24 static ASL letters: A-I, K-Y.