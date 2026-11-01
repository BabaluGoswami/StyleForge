# StyleForge — Neural Style Transfer Studio

Transform ordinary photographs into artistic compositions with **StyleForge**, a web-based Neural Style Transfer application.

StyleForge allows users to upload a content image and a style reference image, adjust the style strength, and generate an artwork that combines visual elements from both images.

**Live Demo:** https://styleforge-7xgt.onrender.com

---

## Overview

Neural Style Transfer is a deep learning technique that combines the content of one image with the artistic style of another.

StyleForge provides a simple web interface for experimenting with this technique without requiring users to run machine learning scripts manually.

The application uses a pretrained neural network model to perform image transformation and provides a downloadable output.

## Features

- **Content Image Upload:** Upload the photograph you want to transform.
- **Style Image Upload:** Select an artistic reference image.
- **Adjustable Style Strength:** Control how strongly the reference style influences the output.
- **Neural Style Transfer:** Generate artwork using a pretrained deep learning model.
- **Image Preview:** View the generated result directly in the browser.
- **Download Artwork:** Download the generated image.
- **Responsive Interface:** Use the application across different screen sizes.
- **Input Validation:** Validate uploaded images and file formats.

## Live Application

<https://styleforge-7xgt.onrender.com>

> Note: The application is hosted on Render's free instance. Initial requests may take longer because the service can spin down after inactivity. Processing time also depends on available server resources.

## Tech Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| Deep Learning | PyTorch, Torchvision |
| Image Processing | Pillow, NumPy |
| Frontend | HTML, CSS, Bootstrap |
| Server | Gunicorn |
| Deployment | Render |

## How It Works

StyleForge follows a Neural Style Transfer workflow:

1. The user uploads a content image.
2. The user uploads a style reference image.
3. The application validates and preprocesses both images.
4. The pretrained model extracts and combines content and style representations.
5. The style strength parameter controls the transformation.
6. The generated artwork is displayed and made available for download.

## Project Structure

```text
StyleForge/
│
├── NST_Code/
│   ├── app.py
│   ├── vgg_normalised.pth
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   ├── utils/
│   │   ├── models.py
│   │   └── utils.py
│   │
│   ├── examples/
│   │
│   └── experiment/
│       └── final_exp/
│           └── decoder_final.pth
│
├── requirements.txt
├── Procfile
├── README.md
└── .gitignore
```

## Getting Started

Follow these steps to run StyleForge locally.

### Prerequisites

- Python 3.10 or compatible Python version
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/BabaluGoswami/StyleForge.git
```

### 2. Navigate to the Project

```bash
cd StyleForge
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
cd NST_Code
python app.py
```

Open the local URL displayed in your terminal.

## Deployment

StyleForge is deployed using [Render](https://render.com/).

Deployment configuration:

| Setting | Value |
|---|---|
| Runtime | Python |
| Branch | main |
| Root Directory | Blank |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `cd NST_Code && gunicorn --workers 1 --threads 1 --timeout 300 --bind 0.0.0.0:$PORT app:app` |
| Health Check | `/healthz` |

### Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `IMAGE_SIZE` | `128` | Controls model input resolution |
| `TORCH_NUM_THREADS` | `1` | Controls PyTorch CPU threads |
| `PYTHON_VERSION` | `3.12.11` | Python runtime version |

Actual performance depends on the available CPU and memory resources.

## Model and Implementation

The application uses pretrained model weights for neural style transfer.

The deployment repository includes the model files required for inference. Training datasets, research notebooks, and unnecessary development files are excluded to keep the deployment repository focused.

## Limitations

- Output quality depends on the input images, style strength, and model resolution.
- Lower image resolution can reduce fine visual details.
- Processing speed depends on available computational resources.
- The free hosting instance may experience cold starts and inference limitations.

## Future Improvements

- Improve preservation of facial and structural details.
- Enhance output resolution while maintaining efficient inference.
- Optimize CPU-based model execution.
- Improve image preprocessing and postprocessing.
- Add additional artistic controls and output options.
- Improve overall generation speed and reliability.

## Author

**Babalu Goswami**

GitHub: https://github.com/BabaluGoswami

---

<p align="center">
  Built with Python, PyTorch and Flask.
  <br>
  <strong>StyleForge — Turn Images Into Art.</strong>
</p>
