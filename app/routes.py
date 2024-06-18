from flask import Blueprint, request, jsonify
import os
import logging
from app.parsers.dxf_parser import parse_dxf

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

    parsed_data = parse_dxf(file_path)
    logger.info(f"Parsed data: {parsed_data}")
    os.remove(file_path)

    return jsonify({"message": f"File {file.filename} parsed successfully", "parsed_data": parsed_data}), 200
