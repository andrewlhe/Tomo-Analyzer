import cv2 as cv
import numpy as np
import os
from sys import platform
from typing import Union
import utilities
from utilities import DiagonalCorners, QuadrilateralCorners


DEBUG = True


def get_corner_preview(image: np.ndarray, corners: QuadrilateralCorners, color: Union[int, tuple[int, int, int]], radius: int, thickness: int) -> np.ndarray:
    result_image = np.array(image)
    for corner in corners.get_list():
        cv.circle(result_image, corner.get_tuple(),
                  radius, color, thickness=thickness)
    return result_image


def get_crop_preview(image: np.ndarray, crop_points: DiagonalCorners, color: Union[int, tuple[int, int, int]], thickness: int) -> np.ndarray:
    result_image = np.array(image)
    cv.rectangle(result_image, crop_points.point1.get_tuple(),
                 crop_points.point2.get_tuple(), color, thickness=thickness)
    return result_image


def get_cropped(image: np.ndarray, crop_points: DiagonalCorners) -> np.ndarray:
    result_image = np.array(image)
    return result_image[crop_points.point1.y:crop_points.point2.y, crop_points.point1.x:crop_points.point2.x]


def process_single_frame(input_directory_path, input_file_name):
    input_file_path = os.path.join(input_directory_path, input_file_name)

    # Read image
    img = cv.imread(input_file_path, flags=cv.IMREAD_ANYDEPTH)
    # cv.imshow("Original Image", img)

    # Normalization
    img_normalized = utilities.get_normalized(img)
    # cv.imshow("Normalized Image", img_normalized)

    # Convert from decimal to integer
    img_8_bit_int = np.round(img_normalized * 255).astype(np.uint8)
    if DEBUG:
        cv.imshow("8-bit Integer Image", img_8_bit_int)

    # Reduce Image Noise
    # # Gaussian Blur
    # img_8_bit_avg = cv.GaussianBlur(img_8_bit_int, (5,5), 0)
    # cv.imshow("Gaussian Blur", img_8_bit_avg)

    # Bilateral
    img_8_bit_bilat = cv.bilateralFilter(img_8_bit_int, 5, 10, 10)
    if DEBUG:
        cv.imshow("Bilateral", img_8_bit_bilat)

    # Edge detection
    canny = cv.Canny(img_8_bit_bilat, 100, 200)
    if DEBUG:
        cv.imshow("Canny Edge Detection", canny)

    # Find the four corners
    corners = utilities.get_corners(canny)
    if DEBUG:
        corner_preview = get_corner_preview(
            img_8_bit_bilat, corners, color=0, radius=4, thickness=2)
        cv.imshow("Corner Preview", corner_preview)

    # Crop
    crop_points = utilities.get_crop_points_with_corners(corners, mode=0)
    if DEBUG:
        crop_preview = get_crop_preview(
            corner_preview, crop_points, color=0, thickness=2)
        cv.imshow("Crop Preview", crop_preview)
    img_cropped = get_cropped(img_8_bit_bilat, crop_points)
    if DEBUG:
        cv.imshow("Cropped", img_cropped)

    # Find Pores

    # References
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 2
    z = img_cropped.reshape((-1, 1))
    z = np.float32(z)
    ret, label, center = cv.kmeans(
        z, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img_cropped.shape)
    if DEBUG:
        cv.imshow("Result", res2)

    # Find Matrix

    # Find Reinforcement

    cv.waitKey(0)


def main():
    input_directory_path = r"Y:\APS\2020-3_1IDC\tomo\32bit\sample_1\hassani_sam1_load0_tomo"
    if platform == "darwin":
        input_directory_path = r"/Users/haoyuanxia/Desktop"
    output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-1_m5\90deg"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".tiff"))]

    for input_file_name in input_file_names:
        process_single_frame(input_directory_path, input_file_name)


if __name__ == "__main__":
    main()
