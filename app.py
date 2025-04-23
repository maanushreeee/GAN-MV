from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import cv2
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mv import (
    load_image,
    apply_cellpose,
    apply_clahe,
    apply_median_blur,
    apply_unsharp_mask,
    apply_color_boost
)

from gan import (generate_images, create_zip_file, MODELS)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Render HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mv')
def mv_page():
    return render_template('mv.html')

@app.route('/gan')
def gan_page():
    return render_template('gan.html')

# Route to handle Machine Vision processing
@app.route('/apply-mv', methods=['POST'])
def apply_mv():
    if 'image' not in request.files or 'technique' not in request.form:
        return jsonify({'error': 'Missing image or technique'}), 400

    file = request.files['image']
    technique = request.form.get('technique')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    try:
        img_rgb = load_image(image_path)

        if technique == 'cellpose':
            result = apply_cellpose(img_rgb)
        elif technique == 'clahe':
            result = apply_clahe(img_rgb)
        elif technique == 'median':
            result = apply_median_blur(img_rgb)
        elif technique == 'unsharp':
            result = apply_unsharp_mask(img_rgb)
        elif technique == 'colorboost':
            result = apply_color_boost(img_rgb)
        else:
            return jsonify({'error': 'Invalid technique'}), 400

        # Convert processed image to PNG bytes
        _, buffer = cv2.imencode('.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/png',
            as_attachment=False,
            download_name='processed.png'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to handle GAN image generation
@app.route('/generate-images', methods=['POST'])
def generate():
    # Get parameters from request
    data = request.get_json()
    cataract_type = data.get('cataractType')
    image_count = data.get('imageCount')
    
    # Validate inputs
    if not cataract_type or not image_count:
        return jsonify({"error": "Missing required parameters"}), 400
    
    if cataract_type not in MODELS:
        return jsonify({"error": "Invalid cataract type"}), 400
    
    try:
        # Generate images
        image_paths = generate_images(cataract_type, image_count)
        
        return jsonify({
            "success": True,
            "message": f"Generated {len(image_paths)} images",
            "images": image_paths
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<cataract_type>')
def download(cataract_type):
    if cataract_type not in MODELS:
        return jsonify({"error": "Invalid cataract type"}), 400
    
    memory_file = create_zip_file(cataract_type)
    return send_file(
        memory_file,
        download_name=f'{cataract_type}_cataract_images.zip',
        as_attachment=True,
        mimetype='application/zip'
    )

# Route to serve generated images (used by GAN)
@app.route('/get-image/<filename>')
def get_image(filename):
    path = os.path.join(GENERATED_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
