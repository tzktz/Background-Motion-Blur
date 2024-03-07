import cv2
from PIL import Image
import numpy as np
from rembg import remove
import os
import shutil
import glob
import moviepy.editor as mp
from moviepy.editor import *


def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def video_to_images(video_path, images_path):
    # Open video
    cam = cv2.VideoCapture(video_path)

    # Get FPS
    fps = cam.get(cv2.CAP_PROP_FPS)

    # Extract audio
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile("./audio.mp3")

    # Create folder for images
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    else:
        shutil.rmtree(images_path)
        os.makedirs(images_path)

    # Go through frames of video
    frameno = 0
    while (True):
        ret, frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = images_path + str(frameno).zfill(5) + '.png'

            print('new frame captured... ', frameno)

            # Save frame
            cv2.imwrite(name, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            frameno += 1
        else:
            break

    # Close video
    cam.release()
    cv2.destroyAllWindows()

    return fps


def images_to_video(images_path, video_export_path, fps):
    # Get a list of PNG images on the "test_images" folder
    images = glob.glob(images_path + "*.png")

    # Sort images by name
    images = sorted(images)

    # Read the first image to get the frame size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    temp_video_path = './temp-video.mp4'

    # Codec
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    # Create final video
    video = cv2.VideoWriter(filename=temp_video_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

    # Read each image and write it to the video
    for i, image in enumerate(images):
        print("Writing frame to video ", i, '/', len(images))

        # Read the image using OpenCV
        frame = cv2.imread(image)

        # Write frame to video
        video.write(frame)

    # Exit the video writer
    video.release()

    # Open final video
    videoclip = VideoFileClip(temp_video_path)

    # Add audio to final video
    audioclip = AudioFileClip("./audio.mp3")
    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip

    # Save final video
    videoclip.write_videofile(video_export_path, audio_codec='aac', codec='libx264')

    # Delete temp files
    os.remove(temp_video_path)
    os.remove("./audio.mp3")


def motion_blur(img, distance, amount):
    # Convert to RGBA
    img = img.convert('RGBA')

    # Convert pil to cv
    cv_img = pil_to_cv(img)

    # Generating the kernel
    kernel_motion_blur = np.zeros((distance, distance))
    kernel_motion_blur[int((distance - 1) / 2), :] = np.ones(distance)
    kernel_motion_blur = kernel_motion_blur / distance

    # Applying the kernel to the input image
    output = cv2.filter2D(cv_img, -1, kernel_motion_blur)

    # Convert cv to pil
    blur_img = cv_to_pil(output).convert('RGBA')

    # Blend the original image and the blur image
    final_img = Image.blend(img, blur_img, amount)

    return final_img


def background_motion_blur(background, distance_blur, amount_blur):
    # Remove background
    subject = remove(background)
    amount_subject = 1

    # Blur the background
    background_blur = motion_blur(background, distance_blur, amount_blur)

    # Put the subject on top of the blur background
    subject_on_blur_background = background_blur.copy()
    subject_on_blur_background.paste(background, (0, 0), subject)

    # Blend the subject and the blur background
    result = Image.blend(background_blur, subject_on_blur_background, amount_subject)

    return result


def video_motion_blur(video_path, export_video_path, distance_blur, amount_blur, amount_subject):
    # Image folder
    images_path = './images/'

    # Convert video to images and save FPS
    fps = video_to_images(video_path, images_path)

    # Create list of images
    image_path_list = glob.glob(images_path + "*.png")

    # Sort images by name
    image_path_list = sorted(image_path_list)

    # Create folder for blur images
    blur_images_path = './blur_images/'
    if not os.path.exists(blur_images_path):
        os.makedirs(blur_images_path)
    else:
        shutil.rmtree(blur_images_path)
        os.makedirs(blur_images_path)

    # Go through image folder
    count = 0
    for filename in image_path_list:
        # Open image an PIL image
        img = Image.open(filename)

        # Motion blur image
        blur_img = background_motion_blur(img, distance_blur, amount_blur, amount_subject)

        # Save blurred image
        blur_img.save(blur_images_path + str(count).zfill(5) + '.png')

        print('motion blur', str(count), '/', len(image_path_list))

        count += 1

    # Convert blurred images to final video
    images_to_video(blur_images_path, export_video_path, fps)

    # Delete temp folders
    shutil.rmtree(images_path)
    shutil.rmtree(blur_images_path)
