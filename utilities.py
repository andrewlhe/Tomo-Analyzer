import math
import numpy as np
import hashlib
from typing import List, Optional, Tuple, Union


class Point:
    """
    This class represents a point in the Cartesian coordinate system, whose coordinate is represented by (x, y).
    """

    def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0):
        """
        Initialize a ``Point`` object with the specified x-coordinate and y-coordinate.

        :param x: the specified x-coordinate of the point.
        :param y: the specified y-coordinate of the point.
        """
        self.x = x
        self.y = y

    def __eq__(self, other: "Point"):
        """
        Check whether this ``Point`` object is equal to the specified ``Point`` object. Two ``Point`` objects are equal
        if they have the same coordinate.

        :param other: the specified Point object to be compared.
        :return: whether this Point object is equal to the specified Point object.
        """
        if other is None or not isinstance(other, Point):
            return False
        return str(self) == str(other)

    def __hash__(self):
        """
        Get the hash of this point.

        :return: the hash of this point.
        """
        return hash(str(self))

    def __lt__(self, other: "Point"):
        """
        Check whether this ``Point`` object is less than the specified ``Point`` object. A ``Point`` object is smaller
        if it has a smaller x-coordinate. If two ``Point`` objects have the same x-coordinate, one is smaller if it has
        a smaller y-coordinate.

        :param other: the specified Point object to be compared.
        :return: whether this Point object is less than the specified Point object.
        """
        if self.x == other.x:
            return self.y < other.y
        else:
            return self.x < other.x

    def __str__(self):
        """
        Get a string representation of this point in the following format:

        (x, y)

        :return: a string representation of this point.
        """
        x_round_0 = round(self.x)
        x_round_2 = round(self.x, 2)
        y_round_0 = round(self.y)
        y_round_2 = round(self.y, 2)
        x_string = str(x_round_0) if x_round_0 == x_round_2 else str(x_round_2)
        y_string = str(y_round_0) if y_round_0 == y_round_2 else str(y_round_2)
        return "({}, {})".format(x_string, y_string)

    def md5_string(self) -> str:
        """
        Get the MD5 hex string of the string representation of this point.

        :return: the MD5 hex string of the string representation of this point.
        """
        return hashlib.md5(str(self).encode()).hexdigest()

    def get_rounded(self) -> "Point":
        return Point(round(self.x), round(self.y))

    def get_offset(self, x: Union[int, float], y: Union[int, float]) -> "Point":
        return Point(self.x + x, self.y + y)

    def get_tuple(self) -> Tuple[Union[int, float], Union[int, float]]:
        return self.x, self.y


class Line:
    """
    This class represents a line segment bounded by two points in the Cartesian coordinate system.
    """

    def __init__(self, p1: Point, p2: Point):
        """
        Initialize a ``Line`` object with the two specified endpoints.

        :param p1: one of the specified endpoints
        :param p2: the other specified endpoint
        """
        # p1 is always the smaller one to make intersected line segments calculation easier
        if p1 == p2:
            raise Exception("p1 and p2 cannot be the same")
        elif p1 < p2:
            self.p1 = p1
            self.p2 = p2
        else:
            self.p2 = p1
            self.p1 = p2

    def __str__(self):
        """
        Get a string representation of this line in the following format:

        <(x1, y1), (x2, y2)>

        :return: a string representation of this line.
        """
        return "<{}, {}>".format(self.p1, self.p2)

    def get_parameters(self) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
        """
        Get a tuple of ``a``, ``b``, and ``c``, where the three numbers are the from the equation of this line in the
        following form:

        ax + by = c

        :return: a tuple of a, b, and c from the equation of this line.
        """
        a = self.p2.y - self.p1.y
        b = self.p1.x - self.p2.x
        c = a * self.p1.x + b * self.p1.y
        return a, b, c

    def contains_point(self, point: Point, precision_threshold: float = 1e-6) -> bool:
        """
        Check whether the specified point lies on this line segment. The two endpoints are included. Due to possible
        precision loss in floating-point calculation, the allowed error is specified as the precision threshold.

        :param point: the specified point to check.
        :param precision_threshold: the allowed error in floating-point calculation.
        :return: whether the specified point lies on this line segment.
        """
        (a, b, c) = self.get_parameters()
        if abs(a * point.x + b * point.y - c) <= precision_threshold:
            if min(self.p1.x, self.p2.x) <= point.x <= max(self.p1.x, self.p2.x) \
                    and min(self.p1.y, self.p2.y) <= point.y <= max(self.p1.y, self.p2.y):
                return True
        return False

    def get_midpoint(self) -> Point:
        """
        Get the midpoint of the line segment confined by the endpoints of this line.

        :return: the midpoint of the line segment
        """
        x = (self.p1.x + self.p2.x) / 2
        y = (self.p1.y + self.p2.y) / 2
        return Point(x, y)

    def get_intersection_with_line(self, other_line: "Line") -> Optional[Point]:
        """
        Get the intersection point of this line with the specified line. If the two lines overlap, only the
        endpoint of the other line that lies on this line is considered as the intersection point.

        :param other_line: the specified line to intersect with this line.
        """
        (self_a, self_b, self_c) = self.get_parameters()
        (other_a, other_b, other_c) = other_line.get_parameters()

        determinant = self_a * other_b - other_a * self_b

        if determinant == 0:
            # Parallel
            return None
        else:
            # Not parallel
            x = (other_b * self_c - self_b * other_c) / determinant
            y = (self_a * other_c - other_a * self_c) / determinant
            return Point(x, y)

    def get_angle(self) -> float:
        """
        Get the angle of this line with respect to a horizontal line. The value is in degree and the range is (-90, 90].

        :return: the angle of this line
        """
        (a, b, c) = self.get_parameters()
        if b == 0:
            return 90
        else:
            return math.atan(a / b) / math.pi * 180


class DiagonalCorners:
    def __init__(self, p1: Point, p2: Point) -> None:
        self.p1 = p1
        self.p2 = p2

    def __str__(self) -> str:
        return "Point 1: {}\nPoint 2: {}".format(self.p1, self.p2)

    def get_list(self) -> List[Point]:
        return [self.p1, self.p2]


class QuadrilateralCorners:
    def __init__(self, top_left: Point, top_right: Point, bottom_left: Point, bottom_right: Point) -> None:
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

    def __str__(self) -> str:
        return "Top-left: {}\nTop-right: {}\nBottom-left: {}\nBottom-right: {}".format(self.top_left, self.top_right,
                                                                                       self.bottom_left,
                                                                                       self.bottom_right)

    def get_list(self) -> List[Point]:
        return [self.top_left, self.top_right, self.bottom_left, self.bottom_right]


class QuadrilateralEdges:
    def __init__(self, top: Line, bottom: Line, left: Line, right: Line) -> None:
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return "Top: {}\nBottom: {}\nLeft: {}\nRight: {}".format(self.top, self.bottom, self.left, self.right)

    def get_list(self) -> List[Line]:
        return [self.top, self.bottom, self.left, self.right]

    def get_tuple(self) -> Tuple[Line, Line, Line, Line]:
        return self.top, self.bottom, self.left, self.right


class Quadrilateral:
    def __init__(self, corners: QuadrilateralCorners) -> None:
        self.corners = corners

    def get_corners(self) -> QuadrilateralCorners:
        return self.corners

    def get_edges(self) -> QuadrilateralEdges:
        top_edge = Line(self.corners.top_left, self.corners.top_right)
        bottom_edge = Line(self.corners.bottom_left, self.corners.bottom_right)
        left_edge = Line(self.corners.top_left, self.corners.bottom_left)
        right_edge = Line(self.corners.top_right, self.corners.bottom_right)
        return QuadrilateralEdges(top_edge, bottom_edge, left_edge, right_edge)

    def get_midlines(self) -> Tuple[Line, Line]:
        top_edge, bottom_edge, left_edge, right_edge = self.get_edges().get_tuple()

        top_midpoint = top_edge.get_midpoint()
        bottom_midpoint = bottom_edge.get_midpoint()
        left_midpoint = left_edge.get_midpoint()
        right_midpoint = right_edge.get_midpoint()

        top_bottom_midline = Line(top_midpoint, bottom_midpoint)
        left_right_midline = Line(left_midpoint, right_midpoint)

        return top_bottom_midline, left_right_midline

    def get_centroid(self) -> Point:
        top_bottom_midline, left_right_midline = self.get_midlines()
        return top_bottom_midline.get_intersection_with_line(left_right_midline)
    
    def get_rotation_angle(self) -> float:
        top_bottom_midline, left_right_midline = self.get_midlines()

        top_bottom_midline_angle = top_bottom_midline.get_angle()
        left_right_midline_angle = left_right_midline.get_angle()

        return (abs(top_bottom_midline_angle + left_right_midline_angle) - 90) / 2

    def get_crop_points(self, mode: int) -> DiagonalCorners:
        # Mode 0: cropped with circumscribed rectangle
        # Mode 1: cropped with inscribed rectangle

        corner_list = self.corners.get_list()

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


def get_normalized(array: np.ndarray, output_min=0.0, output_max=1.0) -> np.ndarray:
    input_min = np.min(array)
    input_max = np.max(array)
    if input_min != output_max:
        scale_factor = (output_max - output_min) / (input_max - input_min)
        return (array - input_min) * scale_factor + output_min
    else:
        return np.array(array)


def get_quadrilateral(binary_2d_array: np.ndarray, target_value=255) -> Quadrilateral:
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

    return Quadrilateral(
        QuadrilateralCorners(top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner))
