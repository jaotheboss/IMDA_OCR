# IMDA AI Scientist Project

## Details
A website uses Captchas on a form to keep the web-bots away. However, the captchas it generates, are quite similar each time:
- the number of characters remains the same each time  
- the font and spacing is the same each time  
- the background and foreground colors and texture, remain largely the same
- there is no skew in the structure of the characters.  
- the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

Delieverables
- `README.md` file
- Python code

Code example:
```{python}
class Captcha(object):
    def __init__(self):
        pass

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        pass
```

## Repository Organisation
1. All model related files are stored under the `src` folder
   1. Within the `src` folder there is a main `classifier.py` file which will collate all of the possible models available for use
   2. The remaining files under `src` will contain model specific code logic
2. All kinds of artifacts are stored under the `artifacts` folder
   1. Data and stored models will reside in their respective folders
   2. Models that are downloaded from the web will not be stored in these folders

## Set-up
1. Create a virtual environment by running: `python3.8 -m venv .env`
2. Activate the virtual environment by running: `source .env/bin/activate`
3. Upgrade `pip` in the virtual environment by running: `pip install --upgrade pip`
4. Download required packages by running: `pip install -r requirements.txt`

## Inference
1. Image paths have already been pre-defined. 
   1. `img_path`: a file path to a single image
   2. `img_list`: a list of image file paths 
2. Output file paths have also been pre-defined
   1. `output_path`: output file path for single output
   2. `output_list`: output file path for multiple outputs
   3. For single outputs, the output file will contain a dictionary with key = predicted value, value = confidence value
   4. For multiple outputs, the output file will contain a dictionary with key = file path of image passed for prediction, value = {predicted value: confidence value}
3. Run the `inference.py` as a script; by running: `python inference.py`
   1. Output files should be seen in the main directory alongside `inference.py`

## Evaluation
1. Declare the json multiple output file and the folder holding the real values
2. Run the evaluation script; by running: `python evaluation.py` 

Note: The inputs and outputs to the general classifier are the same. Referring to `Inference.2.3` and `Inference.2.4` for the I/O standards.

## Models
1. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
   1. Calling the model will trigger a download from the internet. This model will take up space in the virtual environment but would only be downloaded once. 
   2. Internet connection is required for this model to work. Having said that, downloading the model and storing it for loading in the future should not be a difficult task. 
   3. This model has a permissive license, which means that it has free-use even for commercial purposes. 
   4. Size: When downloading off the web, it is 4MB in size. 
   5. Accuracy: 22.92 CER (i.e 1 out of 5 characters wrong on average)

2. [HF](https://huggingface.co/microsoft/trocr-small-printed)
   1. Similar to PaddleOCR, calling the model will trigger a download from the internet. 
   2. Unlike PaddleOCR, the Huggingface (HF) model is not under a permissive license and will therefore require a little more legal intervention to ascertain its eligibility for production usage.
   3. Size: The model is large at 200MB in size.
   4. Accuracy: 27.78 CER (i.e slightly more than 1.25 out of 5 characters wrong on average)