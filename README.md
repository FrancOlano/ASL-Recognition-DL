# ASL Recognition Demo

This folder contains the premium client-side browser demo for the ASL fingerspelling model.

## What it does

- Opens the user camera in the browser.
- Uses ONNX Runtime Web to perform inference locally in your browser.
- Recognizes one static ASL letter or special symbol at a time.
- Appends stable predictions into a text area so the user can spell words.
- Hardware-accelerated (WebGL) inference that guarantees privacy, as frames never leave the user's computer.

## Requirements

The app requires an exported ONNX model (`model.onnx`) and the label map (`classes.json`). These files are loaded directly by the browser. 

## Run it locally

Because modern browsers restrict fetching local files (like `.onnx` and `.json`) via the `file://` protocol due to CORS security rules, you need to serve the files using a simple local web server.

From the repository root, run:

```bash
python demo/app.py
```

Then open `http://127.0.0.1:5000` in a browser and allow camera access.

Alternatively, you can run any static web server in this directory, for example:
```bash
python -m http.server 8000
```