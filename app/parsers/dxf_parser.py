from app.parsers.dxf_parsing_main import initialize

def parse_dxf(file_path):
    page = initialize(file_path)
    return page
