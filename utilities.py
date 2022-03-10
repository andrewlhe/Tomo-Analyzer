from typing import Union


class Point:
    def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return "({}, {})".format(self.x, self.y)

    def get_offset(self, x: Union[int, float], y: Union[int, float]) -> "Point":
        return Point(self.x + x, self.y + y)

    def get_tuple(self) -> tuple[Union[int, float], Union[int, float]]:
        return self.x, self.y


class DiagonalCorners:
    def __init__(self, point1: Point = Point(), point2: Point = Point()) -> None:
        self.point1 = point1
        self.point2 = point2

    def __str__(self) -> str:
        return "Point 1: {}\nPoint 2: {}".format(self.point1, self.point2)

    def get_list(self) -> list[Point]:
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
