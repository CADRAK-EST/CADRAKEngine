from flask import Blueprint, request, jsonify, Response
import os
import logging
from app.parsers.dxf_parser import parse_dxf
import zipfile
import shutil
import json

logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

TEMP_UPLOAD_FOLDER = 'tmp'

if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

@main.route('/parse', methods=['POST'])
def parse_file():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if not (file.filename.endswith('.dxf') or file.filename.endswith('.zip')):
        logger.error("Invalid file type. Only .dxf or .zip files are allowed")
        return jsonify({"error": "Invalid file type. Only .dxf or .zip files are allowed"}), 400

    file_path = os.path.join(TEMP_UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    def generate():
        if file.filename.endswith('.dxf'):
            page = parse_dxf(file_path)
            yield json.dumps(page).encode('utf-8')  # Convert JSON to bytes
        elif file.filename.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                temp_dir = os.path.join(TEMP_UPLOAD_FOLDER, 'extracted')
                zip_ref.extractall(temp_dir)
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.dxf'):
                        dxf_path = os.path.join(temp_dir, file_name)
                        page = parse_dxf(dxf_path)
                        yield json.dumps(page).encode('utf-8')  # Convert JSON to bytes
                        os.remove(dxf_path)
                shutil.rmtree(temp_dir)
        os.remove(file_path)

    return Response(generate(), mimetype='application/json')
