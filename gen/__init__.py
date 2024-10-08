import sys
import signal
import math
import random
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPainter, QPixmap, QColor, QPen
from PyQt6.QtCore import QPointF


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

app = QApplication(sys.argv)


def interpolant(t):
    return ((6.0 * t - 15.0) * t + 10.0) * t * t * t


def generate_perlin_noise_2d(
    shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def draw(
    width,
    height,
):
    seed = random.randint(0, 100000000)

    # Set the random seed for repeatability
    np.random.seed(seed)

    p = QPainter()
    p.image = QPixmap(width, height)
    p.begin(p.image)
    p.image.fill(QColor(255, 255, 255))

    p.setPen(QPen(QColor(0, 0, 0)))

    p_noise = generate_perlin_noise_2d((width, height), (3, 2))

    amount_of_lines = 10000
    max_length = 10
    starting_points = [
        (random.randint(0, width - 1), random.randint(0, height - 1))
        for _ in range(amount_of_lines)
    ]

    for x_start, y_start in starting_points:
        # The current line length tracking variable
        line_length = 0

        # Actually draw the flow field
        while line_length < max_length:
            # angle between -pi and pi
            angle = p_noise[int(x_start), int(y_start)] * math.pi

            # Compute the new point
            x_next = x_start + math.cos(angle)
            y_next = y_start + math.sin(angle)

            # Draw the line
            p.drawLine(QPointF(x_start, y_start), QPointF(x_next, y_next))

            # Update the line length
            line_length += math.sqrt((x_next - x_start) ** 2 + (y_next - y_start) ** 2)

            # Break from the loop if the new point is outside our image bounds
            # or if we've exceeded the line length; otherwise update the point
            if (
                x_next < 0
                or x_next >= width
                or y_next < 0
                or y_next >= height
                or line_length > max_length
            ):
                break
            else:
                x_start, y_start = x_next, y_next

    p.image.save(f"image_{seed}.jpg", "jpg", 100)


while True:
    try:
        draw(3000, 2000)
    except ValueError as e:
        print(e)
