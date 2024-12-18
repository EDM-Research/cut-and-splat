import random

import numpy as np
import open3d as o3d


class Plane:
    """
    Class describing a plane by the normal vector and a point on the plane
    """
    def __init__(self, normal: np.array, center: np.array):
        self.normal = normalize(normal)
        self.center = center

    def equation(self) -> np.array:
        """
        Get the equation form of this plane (a, b, c, d)
        """
        a, b, c = self.normal
        x_c, y_c, z_c = self.center
        d = -(a * x_c + b * y_c + c * z_c)
        return np.array([a, b, c, d])

    def origin(self):
        a, b, c, d = self.equation()

        z_value = 0

        # Solve for x and y
        x = 0  # Setting one variable to zero
        y = (-a * x - c * z_value - d) / b

        # Get the coordinates of the origin on the plane
        origin_on_plane = np.array([x, y, z_value])

        return origin_on_plane

    def project_point(self, point):
        plane_origin = self.origin()
        plane_normal = normalize(self.normal)

        vector_to_point = point - plane_origin

        # Project the vector onto the plane
        projected_vector = vector_to_point - np.dot(vector_to_point, plane_normal) * plane_normal

        # Add the projected vector to any point on the plane
        projected_point = plane_origin + projected_vector

        return projected_point

    def distance(self, point):
        return np.dot(self.normal, point - self.center)

    def flip_normal(self):
        self.normal *= -1

    @staticmethod
    def from_equation(a, b, c, d):
        magnitude = np.linalg.norm([a, b, c])
        normal_vector = np.array([a, b, c]) / magnitude
        x = 0
        y = 0
        z = -d / c if c != 0 else 0

        return Plane(normal_vector, np.array([x, y, z]))


def vector_angle(a: np.array, b: np.array) -> float:
    """
    Compute the angle between the two vectors
    """
    a = normalize(a)
    b = normalize(b)

    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def align_vectors(a, b):
    """
    https://stackoverflow.com/questions/67017134/find-rotation-matrix-to-align-two-vectors
    """
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def get_translation_matrix(x, y, z):
    # Create a 4x4 identity matrix
    translation_matrix = np.identity(4)

    # Set the translation components
    translation_matrix[0, 3] = x
    translation_matrix[1, 3] = y
    translation_matrix[2, 3] = z

    return translation_matrix


def normalize(vector: np.array):
    return vector / np.linalg.norm(vector)


def sample_sphere(center: np.array, radius: float):
    point = np.random.normal(0.0, 1.0, (3,))
    point = normalize(point)

    return point * radius + center


def rotation_axes(R: np.array):
    normal_axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    rotated = []
    for axis in normal_axes:
        rot = np.dot(R, axis)
        rot /= np.linalg.norm(rot)
        rotated.append(rot)

    return rotated


def normal_of_bbox(bbox: o3d.geometry.OrientedBoundingBox) -> np.array:
    """
    Given an oriented bounding box, compute the normal of that box
    """
    unit_normal = np.array([0, 0, 1])
    normal = np.dot(bbox.R, unit_normal)
    normal /= np.linalg.norm(normal)

    return normal


def count_points_above_plane(plane: Plane, points):
    # Ensure plane_center and plane_normal are numpy arrays
    plane_center = plane.center
    plane_normal = plane.normal / np.linalg.norm(plane.normal)  # Normalize the normal vector

    # Convert points to a numpy array
    points = np.array(points)

    # Calculate the vector from plane_center to each point
    vectors_to_points = points - plane_center

    # Calculate the signed distances from the points to the plane
    distances = np.dot(vectors_to_points, plane_normal)

    # Count the number of points with positive signed distance
    count_above = np.sum(distances > 0)

    return count_above


def find_closest_axis(bbox: o3d.geometry.OrientedBoundingBox, vector: np.array) -> np.array:
    """
    Given an oriented bounding box, find its axis that is closest to the given vector
    """
    axes = rotation_axes(bbox.R)
    axes.extend([-1.0 * a for a in axes])

    distances = [np.linalg.norm(vector - a) for a in axes]
    min_index = np.argmin(distances)

    if min_index > 2:
        min_index -= 3

    return axes[min_index]


def flip_bbox(bbox: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
    bbox.rotate(np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]]))

    return bbox


def rotate_point_around_vector(to_rotate, rotation_center, rotation_axis, theta):
    # Convert theta to radians
    theta = np.radians(theta)

    # Translate point c to the origin relative to p
    c_prime = np.array(to_rotate) - np.array(rotation_center)

    # Normalize the vector v
    rotation_axis = np.array(rotation_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Rodrigues' rotation formula
    c_prime_rot = (c_prime * np.cos(theta) +
                   np.cross(rotation_axis, c_prime) * np.sin(theta) +
                   rotation_axis * np.dot(rotation_axis, c_prime) * (1 - np.cos(theta)))

    # Translate the point back
    c_new = c_prime_rot + np.array(rotation_center)

    return c_new


def calculate_geometric_center(points: np.array):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Calculate the geometric center (midpoint of the bounding box)
    geometric_center = (min_coords + max_coords) / 2

    return geometric_center


def get_reflection_matrix(normal, center):
    normal = normalize(normal)

    # Translation to origin
    T1 = np.identity(4)
    T1[:3, 3] = -np.array(center)

    # Reflection matrix
    reflection = np.identity(4)
    reflection[:3, :3] -= 2 * np.outer(normal, normal)

    # Translation back to original position
    T2 = np.identity(4)
    T2[:3, 3] = np.array(center)

    # Combine the transformations: T2 * reflection * T1
    return T2 @ reflection @ T1


def distances_to_plane(points: np.array, plane: Plane):
    A, B, C = plane.normal
    x0, y0, z0 = plane.center

    # get the distance for each point to the ground plane
    diff = points - np.array([x0, y0, z0])
    numerators = np.abs(A * diff[:, 0] + B * diff[:, 1] + C * diff[:, 2])
    denominator = np.sqrt(A ** 2 + B ** 2 + C ** 2)
    distances = numerators / denominator

    return distances


if __name__ == "__main__":
    import open3d as o3d

    rotation_center = np.array([0, 0.3, 0.7])
    rotation_axis = np.array([1, 1, 0])
    point = np.array([0.2, 0.6, 0.1])

    points = []

    for _ in range(100):
        rotated = rotate_point_around_vector(point, rotation_center, rotation_axis, random.randint(0, 360))
        points.append(rotated)

    points.append(rotation_center)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([rotation_center, rotation_center + rotation_axis * 10]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd, line_set])