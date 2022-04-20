import cv2 as cv
import numpy as np
import os
from sys import platform
from typing import Tuple, Union
import utilities
from utilities import DiagonalCorners, Quadrilateral

DEBUG = True


def get_quadrilateral_preview(image: np.ndarray, quadrilateral: Quadrilateral, color: Union[int, Tuple[int, int, int]],
                              corner_circle_radius: int, thickness: int) -> np.ndarray:
    result_image = np.array(image)

    for corner in quadrilateral.get_corners().get_list():
        cv.circle(result_image, corner.get_tuple(),
                  corner_circle_radius, color, thickness=thickness)

    for edge in quadrilateral.get_edges().get_list():
        cv.line(result_image, edge.p1.get_tuple(),
                edge.p2.get_tuple(), color, thickness=thickness)

    return result_image


def get_centroid_preview(image: np.ndarray, quadrilateral: Quadrilateral, color: Union[int, Tuple[int, int, int]],
                         corner_circle_radius: int, thickness: int) -> np.ndarray:
    result_image = np.array(image)

    top_bottom_midline, left_right_midline = quadrilateral.get_midlines()
    for midline in [top_bottom_midline, left_right_midline]:
        cv.circle(result_image, midline.p1.get_rounded().get_tuple(),
                  corner_circle_radius, color, thickness=thickness)
        cv.circle(result_image, midline.p2.get_rounded().get_tuple(),
                  corner_circle_radius, color, thickness=thickness)
        cv.line(result_image, midline.p1.get_rounded().get_tuple(),
                midline.p2.get_rounded().get_tuple(), color, thickness=thickness)

    centroid = quadrilateral.get_centroid()
    cv.circle(result_image, centroid.get_rounded().get_tuple(),
              corner_circle_radius, color, thickness=thickness)

    return result_image


def get_crop_preview(image: np.ndarray, crop_points: DiagonalCorners, color: Union[int, Tuple[int, int, int]],
                     thickness: int) -> np.ndarray:
    result_image = np.array(image)

    cv.rectangle(result_image, crop_points.p1.get_tuple(),
                 crop_points.p2.get_tuple(), color, thickness=thickness)

    return result_image


def get_cropped(image: np.ndarray, crop_points: DiagonalCorners) -> np.ndarray:
    result_image = np.array(image)
    return result_image[crop_points.p1.y:crop_points.p2.y, crop_points.p1.x:crop_points.p2.x]


def process_single_frame(input_directory_path: str, input_file_name: str) -> None:
    input_file_path = os.path.join(input_directory_path, input_file_name)

    # Read image
    img = cv.imread(input_file_path, flags=cv.IMREAD_ANYDEPTH)
    if DEBUG:
        cv.imshow("Original Image", img)

    # Normalization
    img_normalized = utilities.get_normalized(img)
    if DEBUG:
        cv.imshow("Normalized Image", img_normalized)

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

    # Find the quadrilateral
    quadrilateral = utilities.get_quadrilateral(canny)
    if DEBUG:
        quadrilateral_preview = get_quadrilateral_preview(
            img_8_bit_bilat, quadrilateral, color=0, corner_circle_radius=4, thickness=2)
        cv.imshow("Quadrilateral Preview", quadrilateral_preview)

    # Find the centroid
    centroid = quadrilateral.get_centroid()
    if DEBUG:
        centroid_preview = get_centroid_preview(
            quadrilateral_preview, quadrilateral, color=0, corner_circle_radius=4, thickness=2)
        cv.imshow("Centroid Preview", centroid_preview)

    # Find the rotation angle
    rotation_angle = quadrilateral.get_rotation_angle()
    if DEBUG:
        print("Rotation angle: {} deg".format(rotation_angle))

    # Rotate
    image_height, image_width = img_8_bit_bilat.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D(center=centroid.get_tuple(), angle=rotation_angle, scale=1)
    rotated_image = cv.warpAffine(src=img_8_bit_bilat, M=rotation_matrix, dsize=(image_width, image_height))
    rotated_image_with_visualization = cv.warpAffine(src=centroid_preview, M=rotation_matrix,
                                                     dsize=(image_width, image_height))
    if DEBUG:
        cv.imshow("Rotated Image", rotated_image)
        cv.imshow("Rotated Image with Visualization", rotated_image_with_visualization)

    # Crop
    crop_points = quadrilateral.get_crop_points(mode=0)
    if DEBUG:
        crop_preview = get_crop_preview(
            quadrilateral_preview, crop_points, color=0, thickness=2)
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

    # Pause for debug
    if DEBUG:
        cv.waitKey(0)


def main() -> None:
    input_directory_path = r"Y:\APS\2020-3_1IDC\tomo\32bit\sample_1\hassani_sam1_load0_tomo"
    if platform == "darwin":
        input_directory_path = r"/Users/haoyuanxia/Desktop"
    output_directory_path = r"Y:\APS\2020-3_1IDC\tomo\result\sample_1"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
            os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".tiff"))]

    for input_file_name in input_file_names:
        process_single_frame(input_directory_path, input_file_name)


if __name__ == "__main__":
    main()
