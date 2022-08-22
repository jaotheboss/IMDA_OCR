import json
from PIL import Image
from torch import Tensor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

_PRETRAINED_MODEL = "microsoft/trocr-small-printed"

hf_processor = TrOCRProcessor.from_pretrained(_PRETRAINED_MODEL)
hf_model = VisionEncoderDecoderModel.from_pretrained(_PRETRAINED_MODEL)


class HFModel():
    @staticmethod
    def preprocess_image(im_path: str) -> Tensor:
        """
        Processing and encoding of image
        """
        image = Image.open(im_path)
        pixel_values = hf_processor(
            images=image, return_tensors="pt").pixel_values
        return pixel_values

    @staticmethod
    def decode_result(pred_ids: Tensor) -> str:
        generated_text = hf_processor.batch_decode(
            pred_ids, skip_special_tokens=True)[0]
        return generated_text

    @staticmethod
    def single_inference(im_path: str, save_path: str = "") -> dict:
        encoded_image = HFModel.preprocess_image(im_path)
        generated_ids = hf_model.generate(encoded_image)
        pred_text = HFModel.decode_result(generated_ids)

        text_and_scores = {pred_text: None}

        if save_path != "":
            with open(save_path, "w") as output_file:
                json.dump(
                    text_and_scores,
                    output_file
                )
        return text_and_scores

    @staticmethod
    def multiple_inference(im_paths: list, save_path: str = "") -> dict:
        text_and_scores = {}
        for indiv_img_path in im_paths:
            if isinstance(indiv_img_path, str):
                result = HFModel.single_inference(indiv_img_path)
            else:
                print("{} is rejected. Typing should be string".format(
                    indiv_img_path))

            text_and_scores[indiv_img_path] = result

        if save_path != "":
            with open(save_path, "w") as output_file:
                json.dump(
                    text_and_scores,
                    output_file
                )

        return text_and_scores
