import os
from src.classifier import CaptchaOCR

img_path = 'artifacts/data/input/input00.jpg'
img_list = ["artifacts/data/input" + "/" +
            i for i in os.listdir("artifacts/data/input") if i.split('.')[-1] == "jpg"]
output_path = "outputs/hf/single_output.json"
output_list = "outputs/hf/multiple_output.json"


def inference(model, payload, output_file):
    result = model(payload, output_file)
    return result


if __name__ == "__main__":
    ocr = CaptchaOCR()  # only instantiate the model when the script is run

    inference(ocr, img_path, output_path)
    inference(ocr, img_list, output_list)
