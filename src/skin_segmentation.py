import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os

def rgb_to_ycrcb(image):
    rows, cols, channels = image.shape

    ycrcb_image = np.zeros((rows, cols, channels), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            r, g, b = image[i, j]

            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
            cb = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

            ycrcb_image[i, j] = [y, cr, cb]

    return ycrcb_image.astype(np.uint8)

def skin_segmentation(ycrcb_image):
    lower_skin = np.array([0, 140, 80], dtype=np.uint8)
    upper_skin = np.array([255, 170, 130], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)
    # segmented_skin = cv2.bitwise_and(ycrcb_image, ycrcb_image, mask=skin_mask)

    return skin_mask


def median_filter(segmented_skin, ksize=3):
    filtered_skin = cv2.medianBlur(segmented_skin, ksize)
    return filtered_skin

def draw_mass_center_lines(image, skin_mask):
    # Calculate the geometric moments
    moments = cv2.moments(skin_mask)

    # Calculate the mass center coordinates
    mass_center_x = int(moments["m10"] / moments["m00"])
    mass_center_y = int(moments["m01"] / moments["m00"])

    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw horizontal and vertical lines through the mass center
    colored_image = cv2.line(colored_image, (mass_center_x, 0), (mass_center_x, image.shape[0] - 1), (0, 255, 0), 2)
    colored_image = cv2.line(colored_image, (0, mass_center_y), (image.shape[1] - 1, mass_center_y), (0, 255, 0), 2)

    return colored_image


def main():
    cwd = Path(os.getcwd())
    media_dir = cwd.parent / 'segmentation' / 'media'
    input_image_path = "{}/cropped.jpg".format(media_dir)

    # Read the input image
    image = cv2.imread(input_image_path)

    # Resize the image to 64x64
    # image = cv2.resize(image, (64, 64))

    # Convert the image to YCrCb format
    ycrcb_image = rgb_to_ycrcb(image)

    # Save the converted image
    cv2.imwrite("{}/ycrcb_image.jpg".format(media_dir), ycrcb_image)

    segmented_skin = skin_segmentation(ycrcb_image)

    cv2.imwrite("{}/segmented_skin.jpg".format(media_dir), segmented_skin)

    filtered_skin = median_filter(segmented_skin)

    cv2.imwrite("{}/filtered_skin.jpg".format(media_dir), filtered_skin)

    output_image = filtered_skin

    output_image = draw_mass_center_lines(output_image, filtered_skin)

    cv2.imwrite("{}/center_lines.jpg".format(media_dir), output_image)

    # Open the JPG image
    input_image = Image.open("{}/center_lines.jpg".format(media_dir))

    # Save the image in PPM format
    input_image.save("{}/center_lines.ppm".format(media_dir), "PPM")
    

if __name__ == "__main__":
    main()