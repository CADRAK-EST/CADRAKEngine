import ezdxf
import re

# Define the mapping dictionary
attribute_mapping = {
    "designed by": ["designed by", "created by", "drawn by", "drawn", "designed", "created", "author"],
    "checked by": ["checked by", "checked_by", "verified by", "inspected by", "checked"],
    "checked date": ["checked date", "check date", "inspection date"],
    "approved by": ["approved by", "authorized by", "approved", "eng approved by", "mfg approved by", "manufacturing approved by", "approved by mfg"],
    "approval date": ["approval date", "eng approval date", "engineering approval date", "eng date", "mfg approved date", "manufacturing approved date", "mfg date"],
    "title": ["title", "name", "drawing title"],
    "date": ["date", "creation date", "drawn on", "created on"],
    "revision number": ["revision number", "revision", "revision no", "revision nr", "rev", "rev nr", "rev no"],
    "format": ["format", "sheet size", "size"],
    "company": ["company", "organization", "firm"],
    "scale": ["initial view scale", "view scale", "scale", "viewportscale"],
    "part number": ["part number", "part no", "pn", "item number"],
    "contract number": ["contract number", "contract no", "contract"],
    "design activity": ["design activity", "activity"],
    "file name": ["file name", "filename"],
    "fscm number": ["fscm number", "fscm no"],
    "issued by": ["issued by", "issuer"],
    "issue date": ["issue date", "issued date"],
    "sheet": ["sheet", "sheet number"],
    "sheet name": ["sheet name", "sheetname"],
    "sheet number name": ["sheet number name", "sheetnumber-name"],
    "viewport title": ["viewport title", "viewtitle"],
    "subtitle": ["subtitle", "drawing subtitle", "dwg subtitle"],
    "drawing number": ["drawing number", "drawing nr", "drawing no" ,"dwg nr", "dwg no"]
}

def normalize_key(key):
    """Normalize the key by converting to lowercase and removing underscores and spaces."""
    return re.sub(r'[\s_]+', ' ', key.strip().lower())

def standardize_key(key):
    normalized_key = normalize_key(key)
    for standard_key, variations in attribute_mapping.items():
        if normalized_key in map(normalize_key, variations):
            return standard_key
    return key

def parse_dxf_title_block(file_path):
    title_block = {
        "attributes": {},
        "texts": []
    }

    try:
        doc = ezdxf.readfile(file_path)

        # Debugging: Print all block names
        for block in doc.blocks:
            print(f"Block name: {block.name}")

        # Identify the title block by its block name
        for block in doc.blocks:
            if "title" in block.name.lower():
                print(f"Processing block: {block.name}")
                for entity in block:
                    print(f"  Entity: {entity.dxftype()}")
                    if entity.dxftype() == 'ATTDEF':
                        tag = entity.dxf.tag.strip()
                        text = entity.dxf.text.strip()
                        standardized_tag = standardize_key(tag)
                        title_block["attributes"][standardized_tag] = text
                        print(f"    ATTDEF: Tag: {tag}, Text: {text}")

                        # Capture the coordinates of the text
                        text_info = {
                            "tag": standardized_tag,
                            "text": text,
                            "position": (entity.dxf.insert.x, entity.dxf.insert.y)
                        }
                        title_block["texts"].append(text_info)
                        print(f"    TEXT: Text: {text_info['text']}, Position: {text_info['position']}")
                    elif entity.dxftype() == 'MTEXT':
                        text_info = {
                            "text": entity.text.strip(),
                            "position": (entity.dxf.insert.x, entity.dxf.insert.y)
                        }
                        title_block["texts"].append(text_info)
                        print(f"    MTEXT: Text: {text_info['text']}, Position: {text_info['position']}")

        return title_block
    except Exception as e:
        return {"error": str(e)}
