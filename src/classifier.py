from src.paddle_model import PaddleModel
from src.hf_model import HFModel
from typing import Union


class CaptchaOCR(object):
    def __init__(self, model: str = "paddleocr"):
        assert model in [
            "paddleocr",
            "hf"
        ]  # accepted models

        if model == "paddleocr":
            self.predict = PaddleModel.single_inference
            self.predict_batch = PaddleModel.multiple_inference
        elif model == "hf":
            self.predict = HFModel.single_inference
            self.predict_batch = HFModel.multiple_inference
        else:
            self.predict = None
            self.predict_batch = None

    def __call__(self, im_path: Union[list, str], save_path: str):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        if isinstance(im_path, str):
            # inference on a single image; passed a single file path
            return self.predict(im_path, save_path)

        elif isinstance(im_path, list):
            # inference on multiple images; passed a list of file paths
            return self.predict_batch(im_path, save_path)

        else:
            raise TypeError("image path type not recognized")
