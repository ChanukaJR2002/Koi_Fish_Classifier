<<<<<<< HEAD
# Koi Fish Classifier

A deep-learning powered web app that identifies **koi fish varieties** from an uploaded image. The model is built on **EfficientNetB0** (transfer learning) and served through an interactive **Streamlit** interface.

## Project Structure

```
koi-fish-classifier/
│
├── KoiFishClassifier.ipynb       # Training notebook (data preprocessing, model training, evaluation)
├── koi_classifier_app.py         # Streamlit web app for inference
├── koi_efficientnet_model.keras  # Trained EfficientNet model (saved Keras format)
├── Dataset                      # Training images, organized by koi variety (one folder per class)
└── README.md
```

## Koi Varieties Recognized

The classifier is trained to recognize the following 17 koi varieties:

Asagi · Bekko · Doitsu · Ghosiki · Goromo · Hikarimoyo · Hikarimuji · Hikariutsuri · Kawarimono · Kin Gin Rin · Kohaku · Sanke · Showa · Shusui · Tancho · Utsuri · Yamato Nishiki

## Features

- Upload a koi fish image (JPG, PNG, WEBP, BMP) through a clean, dark-themed UI
- Automatic image validation (color space check, minimum 224×224 resolution)
- EfficientNet-based prediction with a confidence score
- Confidence-tiered result feedback (Good / Moderate / Weak confidence)
- Low-confidence detection heuristic to flag likely non-koi images
- Short educational description for each identified koi variety
- Reset/clear functionality for quick re-testing

## Tech Stack

- **Python 3.13.5**
- **TensorFlow / Keras** — EfficientNet model training & inference
- **Streamlit** — web app UI
- **OpenCV (cv2)** & **Pillow (PIL)** — image processing
- **NumPy**

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ChanukaJR2002/Koi_Fish_Classifier.git
cd Koi_Fish_Classifier
```

### 2. Install dependencies
```bash
pip install streamlit tensorflow opencv-python pillow numpy
```

### 3. Run the app
Make sure `koi_efficientnet_model.keras` is in the same directory as `koi_classifier_app.py`, then run:
```bash
streamlit run koi_classifier_app.py
```

The app will open in your browser at `http://localhost:8501`.

## Training

The full training pipeline — data loading, augmentation, EfficientNet transfer learning, and evaluation — is documented in **`KoiFishClassifier.ipynb`**. Open it with Jupyter Notebook/Lab or Google Colab to reproduce or fine-tune the model.

```bash
jupyter notebook KoiFishClassifier.ipynb
```

## Limitations

- Prediction accuracy can degrade with cluttered or unusual backgrounds, poor lighting, or images of fish from atypical angles.
- This application can detect only 17 varities of koi fishes only.
- Uploaded image must include one koi fish only. 

## Repository Contents Checklist (for upload)

- [x] `KoiFishClassifier.ipynb` - training notebook
- [x] `koi_classifier_app.py` - Streamlit inference app
- [x] `koi_efficientnet_model.keras` - trained model file
- [x] `Dataset` - 

## Acknowledgements

Built using transfer learning on EfficientNet, with a custom-curated koi fish image dataset spanning 17 traditional varieties.

## Future Implementations

- Expand the dataset with more images per class to improve accuracy
- Add multi-fish detection — currently it assumes one fish per image; object detection (YOLO) could let it classify multiple koi in a single photo.
- Save users logins and predictions with images (a personal koi history or collection tracker).
- Add a REST API (FastAPI) wrapper around the model so it can be integrated into other apps, not just the Streamlit UI.

## Author Details
---------------
## Name - Chanuka Rajapaksa
## GitHub - [@ChanukaJR2002](https://github.com/)
## Linkedin - https://www.linkedin.com/in/chanuka-rajapaksa-14b9533a1/



