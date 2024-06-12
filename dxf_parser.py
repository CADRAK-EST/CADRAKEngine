from dxf_title_block_parser import parse_dxf_title_block

def parse_dxf(file_path):
    parsed_data = {
        "filename": file_path,
        "title_block": parse_dxf_title_block(file_path)
    }
    return parsed_data
