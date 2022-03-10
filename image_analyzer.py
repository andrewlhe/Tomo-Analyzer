import cv2 as cv
import numpy as np
import os
from typing import Tuple, Union


def get_normalized(array: np.ndarray, output_min=0.0, output_max=1.0) -> np.ndarray:
    input_min = np.min(array)
    input_max = np.max(array)
    if input_min != output_max:
        scale_factor = (output_max - output_min) / (input_max - input_min)
        return (array - input_min) * scale_factor + output_min
    else:
        return np.array(array)


def get_corners(binary_2d_array: np.ndarray, target_value=255) -> Tuple[
        Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    if len(binary_2d_array.shape) != 2:
        raise ValueError("Input array must by 2D")

    rows = binary_2d_array.shape[0]
    cols = binary_2d_array.shape[1]

    top_corner = None
    bottom_corner = None
    left_corner = None
    right_corner = None

    # Get top corner
    for row in range(0, rows, 1):
        for col in range(0, cols, 1):
            if binary_2d_array[row][col] == target_value:
                top_corner = (col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if top_corner is not None:
            # Break from outer loop
            break

    # Get bottom corner
    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            if binary_2d_array[row][col] == target_value:
                bottom_corner = (col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if bottom_corner is not None:
            # Break from outer loop
            break

    # Get left corner
    for col in range(0, cols, 1):
        for row in range(0, rows, 1):
            if binary_2d_array[row][col] == target_value:
                left_corner = (col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if left_corner is not None:
            # Break from outer loop
            break

    # Get right corner
    for col in range(cols - 1, -1, -1):
        for row in range(rows - 1, -1, -1):
            if binary_2d_array[row][col] == target_value:
                right_corner = (col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if right_corner is not None:
            # Break from outer loop
            break

    if top_corner is None or bottom_corner is None or left_corner is None or right_corner is None:
        raise ValueError("The target value is not found in the input")

    return top_corner, bottom_corner, left_corner, right_corner


def get_crop_points_with_corners(corners: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                                 mode: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Mode 0: cropped with circumscribed rectangle
    # Mode 1: cropped with inscribed rectangle

    corner_list = [corner for corner in corners]

    # Sort corners by x
    corner_list.sort(key=lambda x: x[0])
    if mode == 0:
        from_x = corner_list[0][0]
        to_x = corner_list[-1][0]
    else:
        from_x = corner_list[1][0]
        to_x = corner_list[-2][0]

    # Sort corners by y
    corner_list.sort(key=lambda x: x[1])
    if mode == 0:
        from_y = corner_list[0][1]
        to_y = corner_list[-1][1]
    else:
        from_y = corner_list[1][1]
        to_y = corner_list[-2][1]

    return (from_x, from_y), (to_x, to_y)


def get_crop_preview(image: np.ndarray, crop_points: Tuple[Tuple[int, int], Tuple[int, int]],
                     color: Union[int, Tuple[int, int, int]], thickness: int) -> np.ndarray:
    result_image = np.array(image)
    cv.rectangle(result_image, (crop_points[0][0], crop_points[0][1]), (crop_points[1][0], crop_points[1][1]), color,
                 thickness=thickness)
    return result_image


def get_cropped(image: np.ndarray, crop_points: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    result_image = np.array(image)
    return result_image[crop_points[0][1]:crop_points[1][1], crop_points[0][0]:crop_points[1][0]]


def single_frame_processing(input_directory_path, input_file_name):

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
    # cv.imshow('Gaussian Blur', img_8_bit_avg)

    # Bilateral
    img_8_bit_avg = cv.bilateralFilter(img_8_bit_int, 5, 10, 10)
    cv.imshow('Bilateral', img_8_bit_avg)

    cv.waitKey(0)

    # Edge detection
    canny = cv.Canny(img_8_bit_avg, 100, 200)
    cv.imshow("Canny Edge Detection", canny)

    # Find the four corners
    corners = get_corners(canny)
    top_corner, bottom_corner, left_corner, right_corner = corners
    print("Top: {}".format(top_corner))
    print("Botton: {}".format(bottom_corner))
    print("Left: {}".format(left_corner))
    print("Right: {}".format(right_corner))

    # Crop
    crop_points = get_crop_points_with_corners(corners, mode=0)
    crop_preview = get_crop_preview(
        img_8_bit_avg, crop_points, color=0, thickness=1)
    cv.imshow("Crop Preview", crop_preview)
    img_cropped = get_cropped(img_8_bit_avg, crop_points)
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
    cv.imshow('res2', res2)

    # Find Matrix

    # Find Reinforcement

    

    cv.waitKey(0)


def main():

    input_directory_path = "Y://APS//2020-3_1IDC//tomo//32bit//sample_1//hassani_sam1_load0_tomo//"
    output_directory_path = r"Y:\APS\2021-3_1IDC\WAXS_fitting\mapping\He_1-1_m5\90deg"

    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".tiff"))]

    for input_file_name in input_file_names:
        single_frame_processing(input_directory_path, input_file_name)


if __name__ == "__main__":
    main()
