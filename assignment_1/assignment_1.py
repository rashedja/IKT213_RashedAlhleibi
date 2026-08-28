import cv2
import os


def print_image_information(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size = image.size
    data_type = image.dtype

    print("height=", height)
    print("width=", width)
    print("channels=", channels)
    print("size=", size)
    print("data type=", data_type)


def save_camera_info():
    kamera = cv2.VideoCapture(0)

    fps = kamera.get(cv2.CAP_PROP_FPS)
    height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))

    kamera.release()

    mappe_sti = "solutions/"
    os.makedirs(mappe_sti, exist_ok=True)

    fil_sti = mappe_sti + "camera_outputs.txt"
    with open(fil_sti, 'w') as fil:
        fil.write(f"fps= {fps}\n")
        fil.write(f"height= {height}\n")
        fil.write(f"width= {width}\n")

    print("Fil lagret til=", fil_sti)


def main():
    # Oppgave IV
    print("OPPGAVE IV Bildeinformasjon")
    bilde = cv2.imread("iris-1.jpg")

    if bilde is None:
        print("FEIL! Fant ikke iris-1.jpg")
    else:
        print_image_information(bilde)

    print("Webcamera-info")
    save_camera_info()


if __name__ == "__main__":
    main()