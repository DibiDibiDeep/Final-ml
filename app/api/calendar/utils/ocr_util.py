import ast


def convert_to_json(data):
    return [{"coordinates": item[0], "text": item[1]} for item in data]


def normalize_ocr_result(data):
    if type(data) == list:
        return convert_to_json(data)
    else:
        data = ast.literal_eval(data)
        return convert_to_json(data)


# 박스를 y축 기준으로 먼저 그룹화하고, 각 그룹 내에서 x축 기준으로 정렬
def sort_boxes(boxes, y_threshold=20):
    # 1. y축 기준으로 그룹화
    y_groups = {}

    for box in boxes:
        y_coord = box["box"][0][1]  # 박스의 y 좌표

        # 비슷한 y 좌표를 가진 그룹 찾기
        grouped = False
        for group_y in list(y_groups.keys()):
            if abs(group_y - y_coord) <= y_threshold:
                y_groups[group_y].append(box)
                grouped = True
                break

        # 새로운 그룹 생성
        if not grouped:
            y_groups[y_coord] = [box]

    # 2. 각 그룹 내에서 x축 기준으로 정렬하고 결과 합치기
    sorted_boxes = []
    for y_coord in sorted(y_groups.keys()):
        # x축 기준으로 정렬
        group = sorted(y_groups[y_coord], key=lambda x: x["box"][0][0])
        sorted_boxes.extend(group)

    return sorted_boxes
