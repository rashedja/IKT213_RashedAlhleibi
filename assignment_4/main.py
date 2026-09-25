import cv2
import numpy as np
import os


def load_image_or_fail(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def harris_corner_detection(reference_image):
    image = load_image_or_fail(reference_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)

    image[corners > 0.01 * corners.max()] = [0, 0, 255]

    output_file = "harris.png"
    if not cv2.imwrite(output_file, image):
        raise RuntimeError(f"Could not save {output_file}")

    print(f"Created: {output_file}")


def align_images(image_to_align, reference_image, max_features, good_match_precent):
    image = load_image_or_fail(image_to_align)
    reference = load_image_or_fail(reference_image)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(img_gray, None)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

    print(f"SIFT features in image_to_align: {len(kp_img)}")
    print(f"SIFT features in reference_image: {len(kp_ref)}")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des_img, des_ref, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < (good_match_precent * pair[1].distance):
            good_matches.append(pair[0])

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]
    print(f"Good matches used: {len(good_matches)}")

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches were found.")

    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if homography is None:
        raise RuntimeError("Could not calculate homography.")

    h, w = reference.shape[:2]
    aligned = cv2.warpPerspective(image, homography, (w, h))

    inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
    matches_img = cv2.drawMatches(
        image, kp_img,
        reference, kp_ref,
        inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches_img)

    print("Created: aligned.png")
    print("Created: matches.png")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ref_img = os.path.join(script_dir, "reference_img.png")
    target_img = os.path.join(script_dir, "align_this.jpg")

    print("Starting Assignment 4...\n")

    print("Part 1: Harris Corner Detection")
    harris_corner_detection(ref_img)

    print("\nPart 2: Feature-Based Image Alignment")
    print("max_features = 10")
    print("good_match_precent = 0.7\n")

    align_images(target_img, ref_img, max_features=10, good_match_precent=0.7)

    print("\nDone")


if __name__ == "__main__":
    main()