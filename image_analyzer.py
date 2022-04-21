from contextlib import AsyncExitStack
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from sys import platform
from typing import List, Tuple, Union
import utilities
from utilities import DiagonalCorners, Point, Quadrilateral


INPUT_DIRECTORY_PATH = r"Y:\APS\2020-3_1IDC\tomo\32bit\sample_1\hassani_sam1_load0_tomo"
OUTPUT_DIRECTORY_PATH = r"Y:\APS\2020-3_1IDC\tomo\result\sample_1"
if platform == "darwin":
    INPUT_DIRECTORY_PATH = r"/Users/haoyuanxia/Desktop/Input"
    OUTPUT_DIRECTORY_PATH = r"/Users/haoyuanxia/Desktop/Output"

DEBUG = False


def get_quadrilateral_preview(image: np.ndarray, quadrilateral: Quadrilateral, color: Union[int, Tuple[int, int, int]],
                              corner_circle_radius: int, thickness: int) -> np.ndarray:
    result_image = np.array(image)

    for corner in quadrilateral.get_corners().get_list():
        cv.circle(result_image, corner.get_rounded().get_tuple(),
                  corner_circle_radius, color, thickness=thickness)

    for edge in quadrilateral.get_edges().get_list():
        cv.line(result_image, edge.p1.get_rounded().get_tuple(),
                edge.p2.get_rounded().get_tuple(), color, thickness=thickness)

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

    cv.rectangle(result_image, crop_points.p1.get_rounded().get_tuple(),
                 crop_points.p2.get_rounded().get_tuple(), color, thickness=thickness)

    return result_image


def get_cropped_with_points(image: np.ndarray, crop_points: DiagonalCorners) -> np.ndarray:
    result_image = np.array(image)
    return result_image[crop_points.p1.get_rounded().y:crop_points.p2.get_rounded().y,
                        crop_points.p1.get_rounded().x:crop_points.p2.get_rounded().x]


def get_cropped_with_offset(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    result_image = np.array(image)

    image_height, image_width = result_image.shape[:2]
    if top + bottom >= image_height or left + right >= image_width:
        raise ValueError("Invalid offset value(s)")

    return result_image[top:image_height - bottom, left:image_width - right]


def get_padded(image: np.ndarray, top: int, bottom: int, left: int, right: int,
               color: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    result_image = cv.copyMakeBorder(
        image, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
    return result_image


def get_rotated(image: np.ndarray, center: Point, angle: float) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D(
        center=center.get_tuple(), angle=angle, scale=1)
    rotated_image = cv.warpAffine(
        src=image, M=rotation_matrix, dsize=(image_width, image_height))
    return rotated_image


def get_threshold_for_binary_image(image: np.ndarray) -> int:
    # Binary image: an image with only two colors
    return image.min() + (image.max() - image.min()) // 2


def get_proportion_for_binary_image(image: np.ndarray) -> float:
    # Binary image: an image with only two colors
    threshold = get_threshold_for_binary_image(image)
    binary_array = np.where(image > threshold, 1, 0)
    return np.sum(binary_array) / np.size(binary_array)


def process_single_frame(input_file_path: str, output_directory_path: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    # Read image
    image = cv.imread(input_file_path, flags=cv.IMREAD_ANYDEPTH)
    if DEBUG:
        cv.imshow("Original Image", image)

    # Normalization
    image_normalized = utilities.get_normalized(image)
    if DEBUG:
        cv.imshow("Normalized Image", image_normalized)

    # Convert from decimal to integer
    image_8_bit_int = np.round(image_normalized * 255).astype(np.uint8)
    if DEBUG:
        cv.imshow("8-bit Integer Image", image_8_bit_int)

    # Reduce Image Noise
    # # Gaussian Blur
    # image_8_bit_avg = cv.GaussianBlur(image_8_bit_int, (5,5), 0)
    # cv.imshow("Gaussian Blur", image_8_bit_avg)

    # Bilateral
    image_8_bit_bilat = cv.bilateralFilter(image_8_bit_int, 5, 10, 10)
    if DEBUG:
        cv.imshow("Bilateral", image_8_bit_bilat)

    # Edge detection
    canny = cv.Canny(image_8_bit_bilat, 100, 200)
    if DEBUG:
        cv.imshow("Canny Edge Detection", canny)

    # Find the quadrilateral
    quadrilateral = utilities.get_quadrilateral(canny)
    if DEBUG:
        quadrilateral_preview = get_quadrilateral_preview(
            image_8_bit_bilat, quadrilateral, color=64, corner_circle_radius=4, thickness=2)
        cv.imshow("Quadrilateral Preview", quadrilateral_preview)

    # Find the centroid
    centroid = quadrilateral.get_centroid()
    if DEBUG:
        centroid_preview = get_centroid_preview(
            quadrilateral_preview, quadrilateral, color=64, corner_circle_radius=4, thickness=2)
        cv.imshow("Centroid Preview", centroid_preview)

    # Find the rotation angle
    rotation_angle = quadrilateral.get_rotation_angle()
    if DEBUG:
        print("Rotation angle: {} deg".format(rotation_angle))

    # Rotate
    rotated_image = get_rotated(image_8_bit_bilat, centroid, rotation_angle)
    if DEBUG:
        cv.imshow("Rotated Image", rotated_image)

        rotated_image_with_visualization = get_rotated(
            centroid_preview, centroid, rotation_angle)
        cv.imshow("Rotated Image with Visualization",
                  rotated_image_with_visualization)

    # Find the quadrilateral after rotation
    # In the coordinate of the image, the x-axis points to the right and the y-axis points to the bottom. The rotation
    # angle needs to be the inverse to get the correct result.
    rotated_quadrilateral = quadrilateral.get_rotated(
        centroid, -rotation_angle)
    if DEBUG:
        rotated_quadrilateral_preview = get_quadrilateral_preview(
            rotated_image, rotated_quadrilateral, color=64, corner_circle_radius=4, thickness=2)
        cv.imshow("Rotated Quadrilateral Preview",
                  rotated_quadrilateral_preview)

    # Crop
    crop_points = rotated_quadrilateral.get_crop_points(mode=1)
    if DEBUG:
        crop_preview = get_crop_preview(
            rotated_quadrilateral_preview, crop_points, color=0, thickness=2)
        cv.imshow("Crop Preview", crop_preview)
    image_cropped = get_cropped_with_points(rotated_image, crop_points)

    crop_offset = 15
    image_cropped = get_cropped_with_offset(
        image_cropped, crop_offset, crop_offset, crop_offset, crop_offset)
    if DEBUG:
        cv.imshow("Cropped", image_cropped)

    # Segmentation

    # Find pores
    _, image_pores = cv.threshold(
        image_cropped, 128, 255, cv.THRESH_BINARY_INV)
    if DEBUG:
        cv.imshow("Pores", image_pores)

    # Find reinforcement
    # References
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 2
    z = image_cropped.reshape((-1, 1))
    z = np.float32(z)
    ret, label, center = cv.kmeans(
        z, k, None, criteria, 100, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image_cropped.shape)
    if DEBUG:
        cv.imshow("Reinforcement and Matrix", res2)

    reinforcement_threshold = get_threshold_for_binary_image(res2)
    _, image_reinforcement_with_pores = cv.threshold(
        res2, reinforcement_threshold, 255, cv.THRESH_BINARY_INV)
    _, image_matrix_with_pores = cv.threshold(
        res2, reinforcement_threshold, 255, cv.THRESH_BINARY)
    if DEBUG:
        cv.imshow("Reinforcement with Pores", image_reinforcement_with_pores)
        cv.imshow("Matrix with Pores", image_matrix_with_pores)

    # Subtract pores
    image_reinforcement = np.bitwise_and(
        np.bitwise_not(image_pores), image_reinforcement_with_pores)
    image_matrix = np.bitwise_and(np.bitwise_not(
        image_pores), image_matrix_with_pores)
    if DEBUG:
        cv.imshow("Reinforcement", image_reinforcement)
        cv.imshow("Matrix", image_matrix)

    # Calculate the proportion of each part
    proportion_pores = get_proportion_for_binary_image(image_pores)
    proportion_reinforcement = get_proportion_for_binary_image(
        image_reinforcement)
    proportion_matrix = get_proportion_for_binary_image(image_matrix)
    if DEBUG:
        print("Pores:         {:.6f}".format(proportion_pores))
        print("Reinforcement: {:.6f}".format(proportion_reinforcement))
        print("Matrix:        {:.6f}".format(proportion_matrix))

    # Reverse the transformations

    # Reverse the cropping
    original_image_height, original_image_width = image_8_bit_bilat.shape[:2]
    top_crop_amount = crop_points.p1.get_rounded().y + crop_offset
    bottom_crop_amount = original_image_height - \
        crop_points.p2.get_rounded().y + crop_offset
    left_crop_amount = crop_points.p1.get_rounded().x + crop_offset
    right_crop_amount = original_image_width - \
        crop_points.p2.get_rounded().x + crop_offset

    image_pores_padded = get_padded(
        image_pores, top_crop_amount, bottom_crop_amount, left_crop_amount, right_crop_amount, color=0)
    image_reinforcement_padded = get_padded(
        image_reinforcement, top_crop_amount, bottom_crop_amount, left_crop_amount, right_crop_amount, color=0)
    image_matrix_padded = get_padded(
        image_matrix, top_crop_amount, bottom_crop_amount, left_crop_amount, right_crop_amount, color=0)

    # Reverse the rotation
    image_pores_rotated = get_rotated(
        image_pores_padded, centroid, -rotation_angle)
    image_reinforcement_rotated = get_rotated(
        image_reinforcement_padded, centroid, -rotation_angle)
    image_matrix_rotated = get_rotated(
        image_matrix_padded, centroid, -rotation_angle)

    if DEBUG:
        cv.imshow("Pores (Original Position)", image_pores_rotated)
        cv.imshow("Reinforcement (Original Position)",
                  image_reinforcement_rotated)
        cv.imshow("Matrix (Original Position)", image_matrix_rotated)

    # Pause for debug
    if DEBUG:
        cv.waitKey(0)

    array_pores = np.where(image_pores_rotated > 128, 1, 0)
    array_reinforcement = np.where(image_reinforcement_rotated > 128, 1, 0)
    array_matrix = np.where(image_matrix_rotated > 128, 1, 0)

    _, file_base_name, _ = get_file_path_components(input_file_path)

    np.savetxt(os.path.join(output_directory_path, "{}_pores.csv".format(file_base_name)), array_pores, fmt="%d",
               delimiter=",")
    np.savetxt(os.path.join(output_directory_path, "{}_reinforcement.csv".format(file_base_name)), array_reinforcement,
               fmt="%d",
               delimiter=",")
    np.savetxt(os.path.join(output_directory_path, "{}_matrix.csv".format(file_base_name)), array_matrix, fmt="%d",
               delimiter=",")

    return array_pores, array_reinforcement, array_matrix, proportion_pores, proportion_reinforcement, proportion_matrix


def get_file_path_components(file_path: str) -> Tuple[str, str, str]:
    directory_path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    dot_index = file_name.rindex(".")
    base_name = file_name[:dot_index]
    extension = file_name[dot_index + 1:]
    return directory_path, base_name, extension


def plot_3d(data: List[np.ndarray], title: str, x_label: str, y_label: str, z_label: str, save_file_path: str) -> None:
    figure = plt.figure()
    axis = figure.add_subplot(projection="3d")

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_zlabel(z_label)

    xs = []
    ys = []
    zs = []

    for z in range(len(data)):
        layer = data[z]
        for y in range(len(layer)):
            for x in range(len(layer[y])):
                if layer[y][x] == 1:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

    axis.scatter(xs, ys, zs, s=0.1, linewidths=0)

    figure.savefig(save_file_path, dpi=600)
    # plt.show()


def main() -> None:
    input_directory_path = os.path.normpath(INPUT_DIRECTORY_PATH)
    input_file_names = [f for f in os.listdir(input_directory_path) if (
        os.path.isfile(os.path.join(input_directory_path, f)) and f.endswith(".tiff"))]
    input_file_names.sort()

    output_directory_path = os.path.normpath(OUTPUT_DIRECTORY_PATH)
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    pores_3d = []
    reinforcement_3d = []
    matrix_3d = []
    proportion_pores_list = []
    proportion_reinforcement_list = []
    proportion_matrix_list = []

    for index, input_file_name in enumerate(input_file_names):
        input_file_path = os.path.join(input_directory_path, input_file_name)
        array_pores, array_reinforcement, array_matrix, proportion_pores, proportion_reinforcement, proportion_matrix = process_single_frame(
            input_file_path, output_directory_path)

        pores_3d.append(array_pores)
        reinforcement_3d.append(array_reinforcement)
        matrix_3d.append(array_matrix)
        proportion_pores_list.append(proportion_pores)
        proportion_reinforcement_list.append(proportion_reinforcement)
        proportion_matrix_list.append(proportion_matrix)

        print("{} ({}/{})".format(input_file_name,
                                  index + 1, len(input_file_names)))

    plot_3d(pores_3d, title="Pores", x_label="X", y_label="Y", z_label="Z",
            save_file_path=os.path.join(output_directory_path, "pores.png"))
    plot_3d(reinforcement_3d, title="Reinforcement", x_label="X", y_label="Y", z_label="Z",
            save_file_path=os.path.join(output_directory_path, "reinforcement.png"))
    plot_3d(matrix_3d, title="Matrix", x_label="X", y_label="Y", z_label="Z",
            save_file_path=os.path.join(output_directory_path, "matrix.png"))


if __name__ == "__main__":
    main()
