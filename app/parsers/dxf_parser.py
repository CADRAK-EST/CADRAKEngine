from app.parsers.dxf_parsing_main import initialize


def parse_dxf(file_path):
    page = initialize(file_path)  # Default values: visualize=False, save=False, analyze=True, log_times=False
    return page

# parse_dxf("E:\\dev\\CADRAK2\\CADRAKEngine\\app\\parsers\\LauriToru.dxf")
