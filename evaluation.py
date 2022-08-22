import fastwer
import json
import os
import re

predicted_fp = "outputs/hf/multiple_output.json"
actual_fp = "artifacts/data/output"


def extract_file_name(input_fp: str) -> str:
    ext_swap = os.path.splitext(input_fp)[0] + ".txt"
    file_name = ext_swap.split("/")[-1]
    return re.sub("input", "output", file_name)


def get_pred_value(pred_dict: dict) -> str:
    return [key for key, _ in pred_dict.items()][0]


if __name__ == "__main__":
    with open(predicted_fp, "r") as pred_file:
        pred_values = json.load(pred_file)

    cer_values = []

    for fp, pred in pred_values.items():
        file_path = os.path.join(actual_fp, extract_file_name(fp))
        if os.path.exists(file_path):
            pred_value = get_pred_value(pred)

            with open(file_path, "r") as actual_file:
                actual_value = actual_file.readline()
            cer = fastwer.score_sent(pred_value, actual_value, char_level=True)
            cer_values.append(cer)
            print("Pred: {}, Actual: {}, CER: {}".format(
                pred_value, actual_value, cer))
        else:
            print(
                "file path does not exist: {}".format(file_path),
                "/nsupposed to match with input file path: {}".format(fp)
            )

    print("Average CER: {}".format(sum(cer_values)/len(cer_values)))
