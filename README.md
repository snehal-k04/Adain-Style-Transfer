# Neural Style Transfer with AdaIN

Apply the artistic style of any painting or sketch to any photo. This project implements **Adaptive Instance Normalization (AdaIN)** in PyTorch and serves it through a Flask web app.

**Live demo:<https://adain-style-transfer-62nb.onrender.com/>
> The demo runs on a free Render instance. It sleeps when idle, so the first visit can take a minute to wake up, and each stylization takes a while on the free CPU.

## How it works

1. A pretrained **VGG encoder** turns the content image and the style image into feature maps.
2. **AdaIN** shifts the mean and variance of the content features to match those of the style features.
3. A trained **decoder** converts the combined features back into an image.

Because the style is applied by matching feature statistics, one trained model can handle styles it has never seen, in a single forward pass.

## Features

- Upload a content image and a style image, then get the stylized result in the browser
- Works with arbitrary styles, with no retraining per style
- Flask + Flask-WTF web interface with Bootstrap styling and a loading screen while the image is processed

## Datasets

- **Content images:** MS-COCO
- **Style images:** Painter by Numbers

The decoder was trained on Kaggle and Google Colab.

## Tech stack

Python, PyTorch, torchvision, Flask, Flask-WTF, Bootstrap, Pillow, Gunicorn

## Run it locally

```bash
git clone https://github.com/snehal-k04/Adain-Style-Transfer.git
cd Adain-Style-Transfer
py -3.10 -m venv venv          # Windows; use python3.10 on macOS/Linux
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

For local runs the app falls back to a development secret key. For deployment, set a `SECRET_KEY` environment variable.

## Project structure

```
app.py              Flask app
train.py            Training script for the decoder
utils/              Model definitions (VGG encoder, decoder, AdaIN)
templates/          HTML templates
static/             CSS, uploads and outputs
experiment/         Saved decoder checkpoints
vgg_normalised.pth  Pretrained VGG weights
```

## Deployment

Deployed on Render with Gunicorn:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --workers 1 --timeout 120`
- Environment variables: `PYTHON_VERSION=3.10.13`, `SECRET_KEY=<--------->`

## Reference

Huang and Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017).
