import numpy as np
from typing import List, Tuple, Union


class Point:
    def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return "({}, {})".format(self.x, self.y)

    def get_offset(self, x: Union[int, float], y: Union[int, float]) -> "Point":
        return Point(self.x + x, self.y + y)

    def get_tuple(self) -> Tuple[Union[int, float], Union[int, float]]:
        return self.x, self.y


class DiagonalCorners:
    def __init__(self, point1: Point = Point(), point2: Point = Point()) -> None:
        self.point1 = point1
        self.point2 = point2

    def __str__(self) -> str:
        return "Point 1: {}\nPoint 2: {}".format(self.point1, self.point2)

    def get_list(self) -> List[Point]:
        return [self.point1, self.point2]


class QuadrilateralCorners:
    def __init__(self, top_left: Point = Point(), top_right: Point = Point(), bottom_left: Point = Point(), bottom_right: Point = Point()) -> None:
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

    def __str__(self) -> str:
        return "Top-left: {}\nTop-right: {}\nBottom-left: {}\nBottom-right: {}".format(self.top_left, self.top_right, self.bottom_left, self.bottom_right)

    def get_list(self) -> list[Point]:
        return [self.top_left, self.top_right, self.bottom_left, self.bottom_right]


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

    # Get top-left corner
    # Search diagonally
    # / / /
    # / /
    # /
    # col + row = k
    # k: [0, cols - 1]
    # row: [0, k]
    for k in range(0, cols - 1 + 1, 1):
        for row in range(0, k + 1, 1):
            col = k - row
            if row < 0 or col < 0 or row >= rows or col >= cols:
                # Out of range
                break

            if binary_2d_array[row][col] == target_value:
                top_left_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if top_left_corner is not None:
            # Break from outer loop
            break

    # Get top-right corner
    # Search diagonally
    # \ \ \
    #   \ \
    #     \
    # row + k = col
    # k: [cols - 1, 0]
    # row [0, cols - k - 1]
    for k in range(cols - 1, -1, -1):
        for row in range(0, cols - k - 1 + 1, 1):
            col = k + row
            if row < 0 or col < 0 or row >= rows or col >= cols:
                # Out of range
                break

            if binary_2d_array[row][col] == target_value:
                top_right_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if top_right_corner is not None:
            # Break from outer loop
            break

    # Get bottom-left corner
    # Search diagonally
    # \
    # \ \
    # \ \ \
    # col + k = row
    # k: [rows - 1, rows - cols]
    # row [rows - 1, k]
    for k in range(rows - 1, rows - cols - 1, -1):
        for row in range(rows - 1, k - 1, -1):
            col = -k + row
            if row < 0 or col < 0 or row >= rows or col >= cols:
                # Out of range
                break

            if binary_2d_array[row][col] == target_value:
                bottom_left_corner = Point(col, row)  # (x, y) = (col, row)
                # Break from inner loop
                break
        if bottom_left_corner is not None:
            # Break from outer loop
            break

    # Get bottom-right corner
    # Search diagonally
    #     /
    #   / /
    # / / /
    # col + row = k
    # k: [rows + cols - 2, rows - 1]
    # row: [rows - 1, k - cols + 1]
    for k in range(rows + cols - 2, rows - 1 - 1, -1):
        for row in range(rows - 1, k - cols + 1 - 1, -1):
            col = k - row
            if row < 0 or col < 0 or row >= rows or col >= cols:
                # Out of range
                break

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

    corner_list = corners.get_list()

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
    