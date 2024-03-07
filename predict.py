import cv2
from PIL import Image
import numpy as np
from rembg import remove
import os
import shutil
import glob
import moviepy.editor as mp
from moviepy.editor import *
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    @staticmethod
    def cv_to_pil(img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    @staticmethod
    def pil_to_cv(img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

    @staticmethod
    def motion_blur(img, distance, amount):
        img = img.convert('RGBA')
        # Convert pil to cv
        cv_img = Predictor.pil_to_cv(img)
        # Generating the kernel
        kernel_motion_blur = np.zeros((distance, distance))
        kernel_motion_blur[int((distance - 1) / 2), :] = np.ones(distance)
        kernel_motion_blur = kernel_motion_blur / distance
        # Applying the kernel to the input image
        output = cv2.filter2D(cv_img, -1, kernel_motion_blur)
        # Convert cv to pil
        blur_img = Predictor.cv_to_pil(output).convert('RGBA')
        # Blend the original image and the blur image
        final_img = Image.blend(img, blur_img, amount)
        return final_img

    @staticmethod
    def background_motion_blur(background, distance_blur, amount_blur):
        # Remove background
        background = Image.open(background)

        subject = remove(background)
        amount_subject = 1
        # Blur the background
        background_blur = Predictor.motion_blur(background, distance_blur, amount_blur)
        # Put the subject on top of the blur background
        subject_on_blur_background = background_blur.copy()
        subject_on_blur_background.paste(background, (0, 0), subject)
        # Blend the subject and the blur background
        result = Image.blend(background_blur, subject_on_blur_background, amount_subject)
        output_path = "out.png"
        result.save(output_path)
        return output_path

    @staticmethod
    def predict(
            image: Path = Input(description="Input image", default=None),
            distance_blur: int = Input(description="Blur distance",
                                       ge=0,
                                       le=500,
                                       default=200
                                       ),
            amount_blur: int = Input(description="Blur amount",
                                     ge=0,
                                     le=1,
                                     default=1)
    ) -> Path:
        output_path = Predictor.background_motion_blur(image, distance_blur, amount_blur)
        return Path(output_path)



