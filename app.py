from flask import Flask, request, jsonify
from dxf_parser import parse_dxf
import os

app = Flask(__name__)

TEMP_UPLOAD_FOLDER = 'tmp'

if not os.path.exists(TEMP_UPLOAD_FOLDER):
    os.makedirs(TEMP_UPLOAD_FOLDER)

@app.route('/parse', methods=['POST'])
def parse_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.dxf'):
        return jsonify({"error": "Invalid file type. Only .dxf files are allowed"}), 400

    file_path = os.path.join(TEMP_UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    parsed_data = parse_dxf(file_path)
    os.remove(file_path)  # Clean up the temporary file

    return jsonify({"message": f"File {file.filename} parsed successfully", "parsed_data": parsed_data}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
