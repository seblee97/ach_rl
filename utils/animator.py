from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import constants


def animate(images: List[np.ndarray], file_name: str, plot_origin: str):
    """Create animation from images.

    Args:
        images: list of numpy arrays to imshow in animation.
        file_name: name of file under which to save animation.
        plot_origin: orientation of numpy array image.
    """
    # add extension to file_name
    file_name = f"{file_name}.{constants.Constants.ANIMATION_FILE_FORMAT}"

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(image):
        ax.imshow(image, origin=plot_origin)

    anim = FuncAnimation(fig, update, frames=images, interval=200)
    anim.save(
        file_name,
        dpi=80,
        writer="imagemagick",
    )
    plt.close()
