from paddleocr import PaddleOCR
import json

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')


class PaddleModel():
    @staticmethod
    def single_inference(im_path: str, save_path: str = "") -> dict:
        results = ocr_model.ocr(im_path)

        text_and_scores = {
            line[1][0]: line[1][1] for line in results
        }

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
                result = PaddleModel.single_inference(indiv_img_path)
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
