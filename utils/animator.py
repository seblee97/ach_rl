from typing import List

import constants
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate(
    images: List[np.ndarray],
    file_name: str,
    plot_origin: str,
    library: str,
    file_format: str,
):
    """Create animation from images.

    Args:
        images: list of numpy arrays to imshow in animation.
        file_name: name of file under which to save animation.
        plot_origin: orientation of numpy array image.
        library: method for generating video.
    """
    # add extension to file_name
    file_name = f"{file_name}.{file_format}"

    if library == constants.Constants.MATPLOTLIB_ANIMATION:
        fig, ax = plt.subplots(figsize=(10, 10))

        def update(image):
            ax.imshow(image, origin=plot_origin)

        anim = FuncAnimation(fig, update, frames=images, interval=200)
        anim.save(
            file_name,
            dpi=50,
            writer="imagemagick",
        )
        plt.close()
    elif library == constants.Constants.IMAGEIO:
        imageio.mimwrite(file_name, images)
