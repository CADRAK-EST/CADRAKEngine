from app.parsers.dxf_parsing_main import initialize
import os
import zipfile
import json

def parse_file(file):
    TEMP_UPLOAD_FOLDER = 'tmp'
    if not os.path.exists(TEMP_UPLOAD_FOLDER):
        os.makedirs(TEMP_UPLOAD_FOLDER)

    file_path = os.path.join(TEMP_UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    def generate():
        report_card= []
        if file.filename.endswith('.dxf'):
            page = initialize(file_path)
            page["metadata"]["page_number"] = 1
            report_card.append({
                "page_number": page["metadata"]["page_number"],
                "page_type": page["metadata"]["page_type"],
                "dimension_mistakes_count": page["metadata"]["dimension_mistakes_count"],
                "other_mistakes_count": page["metadata"]["other_mistakes_count"]
            })
            yield json.dumps(page).encode('utf-8')
        elif file.filename.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                temp_dir = os.path.join(TEMP_UPLOAD_FOLDER, 'extracted')
                zip_ref.extractall(temp_dir)
                page_number = 1
                for file_name in zip_ref.namelist():
                    if file_name.endswith('.dxf'):
                        dxf_path = os.path.join(temp_dir, file_name)
                        page = initialize(dxf_path)
                        page["metadata"]["page_number"] = page_number
                        report_card.append({
                            "page_number": page["metadata"]["page_number"],
                            "page_type": page["metadata"]["page_type"],
                            "dimension_mistakes_count": page["metadata"]["dimension_mistakes_count"],
                            "other_mistakes_count": page["metadata"]["other_mistakes_count"]
                        })
                        yield json.dumps(page).encode('utf-8')
                        page_number += 1
        yield json.dumps(report_card).encode('utf-8')

    return generate
