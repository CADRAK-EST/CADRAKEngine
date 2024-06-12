from flask import Flask, request, jsonify
from dxf_parser import parse_dxf

app = Flask(__name__)

@app.route('/parse', methods=['POST'])
def parse_file():
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"error": "No file path provided"}), 400

    file_path = data['file_path']
    parsed_data = parse_dxf(file_path)
    return jsonify({"message": f"File {file_path} parsed successfully", "parsed_data": parsed_data}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
