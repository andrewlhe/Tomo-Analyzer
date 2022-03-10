import cv2 as cv
import numpy as np
import os
from typing import Tuple, Union
from utilities import Point, DiagonalCorners, QuadrilateralCorners


def get_normalized(array: np.ndarray, output_min=0.0, output_max=1.0) -> np.ndarray:
    input_min = np.min(array)
    input_max = np.max(array)
    if input_min != output_max:
        scale_factor = (output_max - output_min) / (input_max - input_min)
        return (array - input_min) * scale_factor + output_min
    else:
        return np.array(array)


def get_corners(binary_2d_array: np.ndarray, target_value=255) -> QuadrilateralCorners:
    if len(binary_2d_array.shape) != 2:
        raise ValueError("Input array must by 2D")

    rows = binary_2d_array.shape[0]
    cols = binary_2d_array.shape[1]

    top_left_corner = None
    top_right_corner = None
    bottom_left_corner = None
    bottom_right_corner = None

    # Get top-right corner
    for row in range(0, rows, 1):
        for col in range(0, cols, 1):
            if binary_2d_array[row][col] == target_value:
                top_right_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if top_right_corner is not None:
            # Break from outer loop
            break

    # Get bottom-left corner
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if binary_2d_array[row][col] == target_value:
                bottom_left_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if bottom_left_corner is not None:
            # Break from outer loop
            break

    # Get top-left corner
    for col in range(0, cols, 1):
        for row in range(0, rows, 1):
            if binary_2d_array[row][col] == target_value:
                top_left_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if top_left_corner is not None:
            # Break from outer loop
            break

    # Get bottom-right corner
    for col in range(cols - 1, -1, -1):
        for row in range(rows - 1, -1, -1):
            if binary_2d_array[row][col] == target_value:
                bottom_right_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if bottom_right_corner is not None:
            # Break from outer loop
            break

    if top_left_corner is None or top_right_corner is None or bottom_left_corner is None or bottom_right_corner is None:
        raise ValueError("The target value is not found in the input")

    return QuadrilateralCorners(top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)


def get_crop_points_with_corners(corners: QuadrilateralCorners, mode: int) -> DiagonalCorners:
    # Mode 0: cropped with circumscribed rectangle
    # Mode 1: cropped with inscribed rectangle

    corner_list = [
        corners.top_left,
        corners.top_right,
        corners.bottom_left,
        corners.bottom_right
    ]

    from_point = Point()
    to_point = Point()

    # Sort corners by x
    corner_list.sort(key=lambda point: point.x)
    if mode == 0:
        from_point.x = corner_list[0].x
        to_point.x = corner_list[-1].x
    else:
        from_point.x = corner_list[1].x
        to_point.x = corner_list[-2].x

    # Sort corners by y
    corner_list.sort(key=lambda point: point.y)
    if mode == 0:
        from_point.y = corner_list[0].y
        to_point.y = corner_list[-1].y
    else:
        from_point.y = corner_list[1].y
        to_point.y = corner_list[-2].y

    return DiagonalCorners(from_point, to_point)


def get_crop_preview(image: np.ndarray, crop_points: DiagonalCorners, color: Union[int, Tuple[int, int, int]], thickness: int) -> np.ndarray:
    result_image = np.array(image)
    cv.rectangle(result_image, crop_points.point1.get_tuple(), crop_points.point2.get_tuple(), color, thickness=thickness)
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
    img_normalized = get_normalized(img)
    # cv.imshow("Normalized Image", img_normalized)

    # Convert from decimal to integer
    img_8_bit_int = np.round(img_normalized * 255).astype(np.uint8)
    cv.imshow("8-bit Integer Image", img_8_bit_int)

    # Reduce Image Noise
    # # Gaussian Blur
    # img_8_bit_avg = cv.GaussianBlur(img_8_bit_int, (5,5), 0)
    # cv.imshow("Gaussian Blur", img_8_bit_avg)

    # Bilateral
    img_8_bit_bilat = cv.bilateralFilter(img_8_bit_int, 5, 10, 10)
    cv.imshow("Bilateral", img_8_bit_bilat)

    cv.waitKey(0)

    # Edge detection
    canny = cv.Canny(img_8_bit_bilat, 100, 200)
    cv.imshow("Canny Edge Detection", canny)

    # Find the four corners
    corners = get_corners(canny)
    print(corners)

    # Crop
    crop_points = get_crop_points_with_corners(corners, mode=0)
    crop_preview = get_crop_preview(
        img_8_bit_bilat, crop_points, color=0, thickness=1)
    cv.imshow("Crop Preview", crop_preview)
    img_cropped = get_cropped(img_8_bit_bilat, crop_points)
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
    cv.imshow("Result", res2)

    # Find Matrix

    # Find Reinforcement

    cv.waitKey(0)


def main():
    input_directory_path = r"Y:\APS\2020-3_1IDC\tomo\32bit\sample_1\hassani_sam1_load0_tomo"
    # input_directory_path = r"/Users/haoyuanxia/Desktop"
    output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-1_m5\90deg"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".tiff"))]

    for input_file_name in input_file_names:
        process_single_frame(input_directory_path, input_file_name)


if __name__ == "__main__":
    main()
