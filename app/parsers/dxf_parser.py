from app.parsers.dxf_parsing_main import initialize
import zipfile
import os


def parse_dxf(file_path):
    if file_path.endswith('.dxf'):
        page = initialize(file_path)
        return {"pages": [page]}
    
    elif file_path.endswith('.zip'):
        pages = []
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            temp_dir = os.path.join(os.path.dirname(file_path), 'temp')
            zip_ref.extractall(temp_dir)
            for file in zip_ref.namelist():
                if file.endswith('.dxf'):
                    dxf_path = os.path.join(temp_dir, file)
                    page = initialize(dxf_path)
                    pages.append(page)
                    os.remove(dxf_path)
            os.rmdir(temp_dir)
        return {"pages": pages}

