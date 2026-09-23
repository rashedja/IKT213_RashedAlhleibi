import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def sobel_edge_detection(image):
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    sobel_result = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=1, ksize=1)
    final_sobel = cv2.convertScaleAbs(sobel_result)
    output_path = os.path.join(OUTPUT_DIR, "sobel_lambo.png")
    cv2.imwrite(output_path, final_sobel)
    print(f"Sobel edges saved to: {output_path}")
    return final_sobel


def canny_edge_detection(image, threshold_1, threshold_2):
    gray = _to_gray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    output_path = os.path.join(OUTPUT_DIR, "canny_lambo.png")
    cv2.imwrite(output_path, edges)
    print(f"Canny edges saved to: {output_path}")
    return edges


def template_match(image, template):
    gray_img = _to_gray(image)
    gray_tpl = _to_gray(template)
    result = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)
    result_img = image.copy()
    h, w = gray_tpl.shape
    for pt in zip(*locations[::-1]):
        cv2.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    output_path = os.path.join(OUTPUT_DIR, "template_match.png")
    cv2.imwrite(output_path, result_img)
    print(f"Template match saved to: {output_path}")
    return result_img


def resize(image, scale_factor: int, up_or_down: str):
    if not isinstance(scale_factor, int):
        raise TypeError("scale_factor must be an integer")
    if scale_factor < 1:
        raise ValueError("scale_factor must be an integer >= 1")
    direction = up_or_down.lower()
    if direction not in ("up", "down"):
        raise ValueError("up_or_down must be 'up' or 'down'")
    resized = image.copy()
    for _ in range(scale_factor):
        if direction == "up":
            resized = cv2.pyrUp(resized)
        else:
            resized = cv2.pyrDown(resized)
    output_path = os.path.join(OUTPUT_DIR, f"resize_{direction}.png")
    cv2.imwrite(output_path, resized)
    print(f"Resized ({direction}) saved to: {output_path}")
    return resized


def main():
    lambo_path = os.path.join(SCRIPT_DIR, "lambo.png")
    shapes_path = os.path.join(SCRIPT_DIR, "shapes.png")
    template_path = os.path.join(SCRIPT_DIR, "shapes_template.jpg")

    lambo = cv2.imread(lambo_path, cv2.IMREAD_COLOR)
    shapes = cv2.imread(shapes_path, cv2.IMREAD_COLOR)
    shapes_template = cv2.imread(template_path, cv2.IMREAD_COLOR)

    if lambo is None:
        raise FileNotFoundError(f"Could not load: {lambo_path}")
    if shapes is None:
        raise FileNotFoundError(f"Could not load: {shapes_path}")
    if shapes_template is None:
        raise FileNotFoundError(f"Could not load: {template_path}")

    print("Running Sobel edge detection...")
    sobel_edge_detection(lambo)

    print("Running Canny edge detection...")
    canny_edge_detection(lambo, threshold_1=50, threshold_2=50)

    print("Running template matching...")
    template_match(shapes, shapes_template)

    print("Running resize 'up'...")
    resize(lambo, scale_factor=2, up_or_down="up")

    print("Running resize 'down'...")
    resize(lambo, scale_factor=2, up_or_down="down")

    print(f"\nAll output images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()