import cv2
import yaml
import processing


def ocr(path_to_image):
    rec_numbers = processing.RecognizeNumbers()

    result = []

    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)

    image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    pp_image = processing.PreProcessing(image, params['gamma'], params['contrast'], params['threshold']).result_image
    lines_util = processing.LinesUtil(image, pp_image)

    lines = lines_util.get_lines()

    for line in lines:
        # Name recognition
        name = None
        # todo

        # Surname recognition
        surname = None
        # todo

        index = rec_numbers.get_index(image, line)

        if name or surname or index:
            result.append((name, surname, index))

    return result
