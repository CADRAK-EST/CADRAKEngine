from app.parsers.drawing_parsing import initialize

def parse_dxf(file_path):
    page = initialize(file_path, visualize=False, save=True, analyze=True)
    return page

# parse_dxf("E:\\dev\\CADRAK2\\CADRAKEngine\\app\\parsers\\LauriToru.dxf")
