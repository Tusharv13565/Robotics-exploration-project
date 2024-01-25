import numpy as np

from Functions import line_intersection


class QuadTreeNode:
    def __init__(self, x, y, width, height, depth=0, max_depth=3, max_items=4):
        self.boundary = (x, y, width, height)
        self.lines = []
        self.children = []
        self.depth = depth
        self.max_depth = max_depth
        self.max_items = max_items

    def subdivide(self):
        x, y, width, height = self.boundary
        half_width, half_height = width / 2, height / 2

        self.children = [
            QuadTreeNode(x, y, half_width, half_height, self.depth + 1, self.max_depth, self.max_items),
            QuadTreeNode(x + half_width, y, half_width, half_height, self.depth + 1, self.max_depth, self.max_items),
            QuadTreeNode(x, y + half_height, half_width, half_height, self.depth + 1, self.max_depth, self.max_items),
            QuadTreeNode(x + half_width, y + half_height, half_width, half_height, self.depth + 1, self.max_depth,
                         self.max_items)
        ]

    def contains_point(self, point):
        x, y, width, height = self.boundary
        px, py = point
        return (x <= px < x + width) and (y <= py < y + height)

    def intersects(self, line):
        x, y, width, height = self.boundary
        line_start, line_end = line

        # Check if either endpoint of the line is inside the rectangle
        if self.contains_point(line_start) or self.contains_point(line_end):
            return True

        # Check if the line intersects any of the rectangle's sides
        rectangle_edges = [
            [[x, y], [x + width, y]],
            [[x + width, y], [x + width, y + height]],
            [[x, y + height], [x + width, y + height]],
            [[x, y], [x, y + height]]
        ]

        for edge in rectangle_edges:
            if line_intersection(line, edge):
                return True

        return False

    def intersects_circle(self, circle_center, radius):
        x, y, width, height = self.boundary
        circle_distance_x = abs(circle_center[0] - x - width / 2)
        circle_distance_y = abs(circle_center[1] - y - height / 2)

        if circle_distance_x > (width / 2 + radius) or circle_distance_y > (height / 2 + radius):
            return False

        if circle_distance_x <= (width / 2) or circle_distance_y <= (height / 2):
            return True

        corner_distance_sq = (circle_distance_x - width / 2) ** 2 + (circle_distance_y - height / 2) ** 2

        return corner_distance_sq <= (radius ** 2)


class QuadTree:
    def __init__(self, width, height, max_depth=5, max_items=4):
        self.root = QuadTreeNode(0, 0, width, height, max_depth=max_depth, max_items=max_items)
        self.line_references = {}

    def insert_line(self, line, line_id):
        self.line_references[line_id] = line
        self._insert_line(self.root, line, line_id)

    def _insert_line(self, node, line, line_id):
        # If node at max depth or below max items
        if node.depth == node.max_depth or len(node.lines) < node.max_items:
            node.lines.append(line_id)
            return

        # Otherwise, subdivide and insert line into relevant child nodes
        if not node.children:
            node.subdivide()

        for child in node.children:
            if child.intersects(line):
                self._insert_line(child, line, line_id)

    def query_range(self, circle_center, radius):
        found_line_ids = set()
        self._query_range(self.root, circle_center, radius, found_line_ids)
        return [self.line_references[line_id] for line_id in found_line_ids]

    def _query_range(self, node, circle_center, radius, found_line_ids):
        if node.intersects_circle(circle_center, radius):
            for line_id in node.lines:
                if self.line_within_circle(self.line_references[line_id], circle_center, radius):
                    found_line_ids.add(line_id)

            if node.children:
                for child in node.children:
                    self._query_range(child, circle_center, radius, found_line_ids)

    @staticmethod
    def line_within_circle(line, circle_center, radius):
        x1, y1 = line[0]
        x2, y2 = line[1]
        cx, cy = circle_center

        # Check if either endpoint is inside the circle
        if QuadTree.point_within_circle(x1, y1, cx, cy, radius) or QuadTree.point_within_circle(x2, y2, cx, cy, radius):
            return True

        # Check if the line intersects the circle
        line_length = np.hypot(x2 - x1, y2 - y1)

        distance = abs((x2 - x1) * (y1 - cy) - (x1 - cx) * (y2 - y1)) / line_length

        return distance <= radius

    @staticmethod
    def point_within_circle(x, y, cx, cy, radius):
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
