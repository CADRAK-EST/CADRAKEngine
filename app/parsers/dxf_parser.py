from app.parsers.dxf_parsing_main import initialize


def parse_dxf(file_path):
    page = initialize(file_path)  # initialize(file_path, visualize=False, save=False, analyze=True, log_times=True)
    return page
