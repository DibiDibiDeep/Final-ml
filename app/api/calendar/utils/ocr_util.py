import ast


def convert_to_json(data):
    return [{"coordinates": item[0], "text": item[1]} for item in data]


def normalize_ocr_result(data):
    if type(data) == list:
        return convert_to_json(data)
    else:
        data = ast.literal_eval(data)
        return convert_to_json(data)
