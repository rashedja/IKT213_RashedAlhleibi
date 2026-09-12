import cv2
import numpy as np


def padding(image, border_width):
    padded_image = cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )
    cv2.imwrite('padded_iris.png', padded_image)
    return padded_image


def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite('cropped_iris.png', cropped_image)
    return cropped_image


def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite('resized_iris.png', resized_image)
    return resized_image


def copy(image, emptyPictureArray):
    rows, cols, channels = image.shape

    for row in range(rows):
        for col in range(cols):
            # Kopierer alle farger for hver piksel
            emptyPictureArray[row, col] = image[row, col]

    cv2.imwrite('copied_iris.png', emptyPictureArray)
    return emptyPictureArray


def grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('grayscale_iris.png', gray_image)
    return gray_image


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite('hsv_iris.png', hsv_image)
    return hsv_image


def hue_shifted(image, emptyPictureArray, hue):
    rows, cols, channels = image.shape

    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                original_pixel = int(image[row, col, channel])
                shifted_pixel = (original_pixel + hue) % 256
                emptyPictureArray[row, col, channel] = shifted_pixel

    cv2.imwrite('hue_shifted_iris.png', emptyPictureArray)
    return emptyPictureArray


def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite('smoothed_iris.png', smoothed_image)
    return smoothed_image


def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        filename = 'rotated_90_iris.png'
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        filename = 'rotated_180_iris.png'
    else:
        print(f"Vinkel {rotation_angle} er ikke støttet")
        return image

    cv2.imwrite(filename, rotated_image)
    return rotated_image


def main():
    # Laster inn bildet
    image = cv2.imread('iris.png')

    if image is None:
        raise FileNotFoundError("Kunne ikke finne 'iris.png' bilde. Sørg for at filen ligger i samme mappe.")

    height, width = image.shape[:2]
    print(f"Startet behandling av bilde ({width}x{height})...")

    padding(image, 100)

    x_0 = 200
    x_1 = width - 130
    y_0 = 200
    y_1 = height - 130

    if x_1 > x_0 and y_1 > y_0:
        crop(image, x_0, x_1, y_0, y_1)
    else:
        print("Advarsel: Bildet er for lite for denne beskjæringen.")

    resize(image, 200, 200)

    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copy(image, emptyPictureArray)

    grayscale(image)

    hsv(image)

    empty_shift_array = np.zeros((height, width, 3), dtype=np.uint8)
    hue_shifted(image, empty_shift_array, 50)

    smoothing(image)

    rotation(image, 180)

if __name__ == "__main__":
    main()