from flask import Blueprint, request, jsonify, Response
import os
import logging
from app.parsers.dxf_parser import parse_file

logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

TEMP_UPLOAD_FOLDER = 'tmp'

if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

@main.route('/parse', methods=['POST'])
def parse_file_route():
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

    generate = parse_file(file)

    return Response(generate(), content_type='application/json')
