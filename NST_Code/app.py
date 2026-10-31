
import os
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import (
    Flask,
    render_template,
    send_from_directory,
    request,
    Response
)
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired, NumberRange
from werkzeug.utils import secure_filename
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# Project paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
EXAMPLES_DIR = BASE_DIR / "examples"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Supported image formats
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".jfif",
    ".webp", ".bmp", ".gif", ".tif", ".tiff"
}

# Flask configuration
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "local-development-key-change-before-deployment"
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

Bootstrap(app)


# SEO: Robots.txt
@app.route("/robots.txt")
def robots_txt():
    sitemap_url = request.url_root.rstrip("/") + "/sitemap.xml"

    content = (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {sitemap_url}\n"
    )

    return Response(content, mimetype="text/plain")


# SEO: Sitemap
@app.route("/sitemap.xml")
def sitemap_xml():
    base_url = escape(request.url_root.rstrip("/"))

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>{base_url}/</loc>
        <changefreq>weekly</changefreq>
        <priority>1.0</priority>
    </url>
</urlset>"""

    return Response(xml, mimetype="application/xml")


# Device and image resolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "256"))

if IMAGE_SIZE not in (256, 512):
    raise ValueError("IMAGE_SIZE must be 256 or 512")


# Model checkpoint paths
VGG_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

if not VGG_PATH.is_file():
    raise FileNotFoundError(f"VGG checkpoint not found: {VGG_PATH}")

if not DECODER_PATH.is_file():
    raise FileNotFoundError(f"Decoder checkpoint not found: {DECODER_PATH}")


# Upload form
class UploadForm(FlaskForm):
    content = FileField(
        "Content Image",
        validators=[
            FileAllowed(
                ["jpg", "jpeg", "png", "jfif", "webp",
                 "bmp", "gif", "tif", "tiff"],
                "Unsupported image format."
            )
        ]
    )

    style = FileField(
        "Style Image",
        validators=[
            FileAllowed(
                ["jpg", "jpeg", "png", "jfif", "webp",
                 "bmp", "gif", "tif", "tiff"],
                "Unsupported image format."
            )
        ]
    )

    content_path = HiddenField()
    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0,
        validators=[
            InputRequired(),
            NumberRange(min=0, max=1)
        ]
    )

    submit = SubmitField("Transfer Style")


# Load models
encoder = VGGEncoder(str(VGG_PATH)).to(device).eval()
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device,
        weights_only=True
    )
)

decoder.eval()

print(f"Model loaded successfully on {device}")
print(f"Image resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")


# Upload image and validate its actual contents
def save_upload(file):
    if not file or not file.filename:
        return None

    original_name = secure_filename(file.filename)
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported format. Use JPG, PNG, WEBP, JFIF, BMP, GIF or TIFF."
        )

    filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename

    try:
        file.save(destination)

        with Image.open(destination) as image:
            image.verify()

    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise ValueError(
            "Invalid or corrupted image. Please upload another image."
        ) from exc

    return filename


# Validate previously uploaded filename
def valid_saved_filename(filename):
    if not filename or Path(filename).name != filename:
        return None

    path = UPLOAD_DIR / filename

    if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
        return filename

    return None


# Open image safely and convert to RGB
def load_image(path):
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.seek(0)
            return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ) as exc:
        raise ValueError(
            "Could not read this image. Please upload a valid image."
        ) from exc


# AdaIN style transfer
def style_transfer(content_image, style_image, alpha):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        content_features = encoder(content_tensor, is_test=True)
        style_features = encoder(style_tensor, is_test=True)

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        output = decoder(stylized_features)

    output = output.squeeze(0).cpu().clamp(0, 1)

    return transforms.ToPILImage()(output)


# Main page
@app.route("/", methods=["GET", "POST"])
def index():
    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            # Reuse saved content image or upload a new one
            if form.content.data and form.content.data.filename:
                content_filename = save_upload(form.content.data)
            else:
                content_filename = valid_saved_filename(
                    form.content_path.data
                )

            # Reuse saved style image or upload a new one
            if form.style.data and form.style.data.filename:
                style_filename = save_upload(form.style.data)
            else:
                style_filename = valid_saved_filename(
                    form.style_path.data
                )

            if not content_filename:
                raise ValueError(
                    "Please upload a valid content image."
                )

            if not style_filename:
                raise ValueError(
                    "Please upload a valid style image."
                )

            alpha = form.alpha.data

            if not 0 <= alpha <= 1:
                raise ValueError(
                    "Style strength must be between 0 and 1."
                )

            content_image = load_image(
                UPLOAD_DIR / content_filename
            )

            style_image = load_image(
                UPLOAD_DIR / style_filename
            )

            # Generate stylized image
            output = style_transfer(
                content_image,
                style_image,
                alpha
            )

            result_filename = f"result_{uuid.uuid4().hex}.png"
            result_path = UPLOAD_DIR / result_filename

            output.save(result_path, format="PNG")

            result_image = result_filename

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Style transfer failed")
            error = (
                "Image processing failed. Please try another image."
            )

    elif form.is_submitted():
        error = (
            "Please check your image files and style strength."
        )

    # Preserve uploaded image references for the next submission
    form.content_path.data = (
        content_filename
        or valid_saved_filename(form.content_path.data)
        or ""
    )

    form.style_path.data = (
        style_filename
        or valid_saved_filename(form.style_path.data)
        or ""
    )

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=form.content_path.data or None,
        style_image=form.style_path.data or None,
        error=error
    )


# Uploaded images
@app.route("/uploads/<path:filename>")
def send_image(filename):
    safe_name = valid_saved_filename(filename)

    if not safe_name:
        return "Image not found", 404

    return send_from_directory(UPLOAD_DIR, safe_name)


# Example gallery images
@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory(EXAMPLES_DIR, filename)


# File size error
@app.errorhandler(413)
def file_too_large(error):
    return "Image size must be under 10 MB.", 413


# Run application
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
