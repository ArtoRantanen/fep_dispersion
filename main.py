import numpy as np
import matplotlib.pyplot as plt


def compute_ray_path(ray_origin, direction_vector, sphere_center, radius):
    # Check if ray intersects with the top half-sphere (oriented parallel to the xy plane)
    a = np.dot(direction_vector, direction_vector)
    b = 2 * np.dot(direction_vector, ray_origin - sphere_center)
    c = np.dot(ray_origin - sphere_center,
               ray_origin - sphere_center) - radius ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant > 0:
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)

        # Only considering the intersection that is closer to the ray_origin
        if 0 < t1 < t2:
            intersection = ray_origin + t1 * direction_vector
        else:
            intersection = ray_origin + t2 * direction_vector

        n1 = 1.3
        n2 = 1.184

        normal_vector = (intersection - sphere_center) / np.linalg.norm(
            intersection - sphere_center)
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        cos_theta_i = -np.dot(normal_vector, direction_vector)
        sin_theta_t2 = (n1 / n2) ** 2 * (1 - cos_theta_i ** 2)
        cos_theta_t = np.sqrt(1 - sin_theta_t2)

        refracted_vector = (n1 / n2) * direction_vector + (
                    n1 / n2 * cos_theta_i - cos_theta_t) * normal_vector
        return intersection, intersection + refracted_vector, refracted_vector
    else:
        return None, None, None


def simulate_damage_dispersion(ax, center, radius):
    pixel_size = 47e-6

    for x in range(10):
        for y in range(10):
            ray_origin = np.array([x, y, 0], dtype=float)
            direction_vector = np.array([0, 0, 1], dtype=float)

            intersection, exit_point, refracted_vector = compute_ray_path(
                ray_origin, direction_vector, center, radius)
            if intersection is not None:
                ax.plot([ray_origin[0], intersection[0]],
                        [ray_origin[1], intersection[1]],
                        [ray_origin[2], intersection[2]], color='b')
                ax.plot([intersection[0], exit_point[0]],
                        [intersection[1], exit_point[1]],
                        [intersection[2], exit_point[2]], color='r')
            else:
                ax.plot(
                    [ray_origin[0], ray_origin[0] + 10 * direction_vector[0]],
                    [ray_origin[1], ray_origin[1] + 10 * direction_vector[1]],
                    [ray_origin[2], ray_origin[2] + 10 * direction_vector[2]],
                    color='b')

    # Drawing the damage area (half-sphere oriented with its flat side up)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(np.pi / 2, np.pi,
                    100)  # Adjusted to draw the lower half circle
    x = center[0] + radius * np.outer(np.sin(v), np.cos(u))
    y = center[1] + radius * np.outer(np.sin(v), np.sin(u))
    z = center[2] + radius * np.outer(np.cos(v), np.ones_like(u))
    ax.plot_surface(x, y, z, color='g', alpha=0.5)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    simulate_damage_dispersion(ax, center=(5, 5, 10), radius=5)
    plt.show()
