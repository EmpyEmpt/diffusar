# TODO: Random discoloration
from skimage.util import random_noise
import numpy as np
import cv2

rnd = {
    'int': np.random.randint,
    'float': np.random.uniform
}


def random_arguments(args):
    data = []
    for i in args:
        val = rnd[i[0].__name__]
        val = val(i[1], i[2])
        data.append(val)
    if len(data) == 1:
        return data[0]
    return data


def apply_ghosting(source: np.ndarray, random_args: bool = True, trail_length: int = 10, trail_decay: float = 0.8, shift_x: int = 5, shift_y: int = 5) -> np.ndarray:
    r"""
    Applies ghosting artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - trail_length : `int`, default: `10`

            Controls the amount of 'ghost' images 

        - trail_decay : `float`, default: `0.8`

            Controls the decay of 'ghost' images

        - shift_x, shift_y : `int`, default: `5`

            Controls the shift of 'ghost' images from original position in pixel term

    Returns
    -------
        - `np.ndarray` Processed image
    """
    if random_args:
        trail_length, trail_decay, shift_x, shift_y = random_arguments(
            [
                [int, 3, 20],
                [float, 0.4, 0.9],
                [int, 3, 20],
                [int, 3, 20]
            ]
        )

    height, width, channels = source.shape
    ghosted_img = np.zeros((height, width, channels), np.float32)

    for i in range(trail_length):
        shift = np.array([shift_x * i, shift_y * i], np.float32)
        shifted = cv2.warpAffine(
            source,
            np.float32(
                [
                    [1, 0, shift[0]],
                    [0, 1, shift[1]]
                ]
            ),
            (width, height)
        )
        ghosted_img += shifted * trail_decay**i

    ghosted_img = ghosted_img / np.max(ghosted_img) * 255

    processed_image = np.uint8(np.clip(ghosted_img, 0, 255))
    return processed_image


def apply_distortion(source: np.ndarray, random_args: bool = True, displacement: int = 10) -> np.ndarray:
    r"""
    Applies distortion artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - displacement : `int`, default: `10`

    Returns
    -------
        - `np.ndarray` Processed image

    """
    if random_args:
        displacement = random_arguments(
            [
                [int, 5, 20]
            ]
        )

    height, width, channels = source.shape
    displacement_map = np.random.uniform(-displacement,
                                         displacement, size=(height, width, 2))
    displacement_map = cv2.GaussianBlur(displacement_map, (5, 5), 0)
    mesh_x, mesh_y = np.meshgrid(np.arange(width), np.arange(height))
    mesh_x = mesh_x.astype(np.float32) + displacement_map[:, :, 0]
    mesh_y = mesh_y.astype(np.float32) + displacement_map[:, :, 1]

    mesh_x = mesh_x.astype(np.float32)
    mesh_y = mesh_y.astype(np.float32)

    processed_image = cv2.remap(
        source, mesh_x, mesh_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return processed_image


def apply_color_banding(source: np.ndarray, random_args: bool = True, levels: int = 8) -> np.ndarray:
    r"""
    Applies color banding artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - levels : `int`, default: `8`

    Returns
    -------
        - `np.ndarray` Processed image

    """

    if random_args:
        levels = random_arguments(
            [
                [int, 8, 32]
            ]
        )

    # height, width, channels = img.shape
    processed_image = np.uint8(
        np.round(source / (256 / levels))
    ) * (256 // levels)

    return processed_image


def __apply_silk_screen_effect(source: np.ndarray, levels: int = 8, threshold: int = 128) -> np.ndarray:
    r"""
    Applies silk screen artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - levels : `int`, default: `8`

        - threshold : `int`, default: `128`

    Returns
    -------
        - `np.ndarray` Processed image

    """
    # height, width, channels = img.shape
    quantized = np.uint8(np.round(source / (256 / levels))) * (256 // levels)
    bw = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
    processed_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return processed_image


def apply_rainbow_effect(source: np.ndarray, random_args: bool = True, strength: int = 10) -> np.ndarray:
    """
    Applies rainbow artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - strength : `int`, default: `10`

    Returns
    -------
        - `np.ndarray` Processed image

    """

    if random_args:
        strength = random_arguments(
            [
                [float, 0, 50],
            ]
        )

    height, width, channels = source.shape

    hue_map = np.zeros((height, width, 3), dtype=np.float32)
    hue_map[:, :, 0] = np.linspace(0, strength**1.5, width)
    hue_map = cv2.GaussianBlur(hue_map, (3, 3), 0)
    hue_map = hue_map * 360 / strength

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] + hue_map[:, :, 0]
    hsv[:, :, 0][hsv[:, :, 0] > 360] = hsv[:, :, 0][hsv[:, :, 0] > 360] - 360

    processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return processed_image


def apply_noise(source: np.ndarray, random_args: bool = True, noise_type='gaussian', mean: float = 0, var: float = 0.001) -> np.ndarray:
    r"""
    Applies rainbow artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - noise_type : `str`, default: `'gaussian'`

            Controls type of noise to be applied

            Can be either `'gaussian'` | `'salt_peppet'` | `'pixels'`

        - mean : `float`, default: `0`

        - var : `float`, default: `0.001`

    Returns
    -------
        - `np.ndarray` Processed image

    """

    if random_args:
        mean, var, minmax = random_arguments(
            [
                [float, -128, 128],
                [float, 0, 0.1],
                [int, 8, 64]
            ]
        )
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'pixels'])

    processed_image = source.copy()
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, var ** 0.5, source.shape)
        processed_image = np.clip(source + noise, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        processed_image = random_noise(source, mode='s&p', amount=0.05)
    elif noise_type == 'pixels':
        processed_image = source + np.random.randint(-minmax, minmax, source.shape).astype(np.uint8)

    return processed_image


def apply_screen_tearing(source: np.ndarray, random_args: bool = True, strength: float = 0.5, direction='horizontal') -> np.ndarray:
    r"""
    Applies rainbow artifact to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - direction : `str`, default: `'horizontal'`

            Controls the direction of the tier

            Can be either `'horizontal'` or `'vertical'`

        - strength : `float`, default: `0.5`

    Returns
    -------
        - `np.ndarray` Processed image

    """

    if random_args:
        strength = random_arguments(
            [
                [float, 0.1, 0.9],
            ]
        )
        direction = np.random.choice(['horizontal', 'vertical'])

    height, width, channels = source.shape
    processed_image = np.zeros(source.shape, dtype=np.uint8)

    if direction == 'horizontal':
        tearing_height = int(strength * height)
        processed_image[
            :height-tearing_height,
            :,
            :
        ] = source[tearing_height:, :, :]

        processed_image[
            height-tearing_height:,
            :,
            :
        ] = source[:tearing_height, :, :]
    elif direction == 'vertical':
        tearing_width = int(strength * width)
        processed_image[
            :,
            :width-tearing_width,
            :
        ] = source[:, tearing_width:, :]

        processed_image[
            :,
            width-tearing_width:,
            :
        ] = source[:, :tearing_width, :]
    else:
        raise ValueError("Invalid direction")

    return processed_image


def apply_compression_artifact(source: np.ndarray, random_args: bool = True, quality: int = 50) -> np.ndarray:
    """
    Applies jpeg compression to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - quality : `int`, default: `50`

    Returns
    -------
        - `np.ndarray` Processed image

    """
    if random_args:
        quality = random_arguments(
            [
                [int, 70, 99],
            ]
        )

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', source, params)
    processed_image = cv2.imdecode(enc, 1)

    return processed_image


def apply_moire_pattern(source: np.ndarray, random_args: bool = True, strength: int = 50, wavelength: int = 100, angle: int = 45, grid_size: int = 10) -> np.ndarray:
    """
    Applies moire pattern to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - strength : `int`, default: `50`

        - wavelength : `int`, default: `100`

        - angle : `int`, default: `45`

        - grid_size : `int`, default: `10`

    Returns
    -------
        - `np.ndarray` Processed image

    """
    if random_args:
        strength, wavelength, angle, grid_size = random_arguments(
            [
                [int, 100, 200],
                [int, 50, 200],
                [int, 0, 90],
                [int, 50, 500]
            ]
        )

    height, width = source.shape[:2]

    x, y = np.meshgrid(
        np.linspace(0, width, grid_size),
        np.linspace(0, height, grid_size)
    )

    angle_radians = np.radians(angle)
    x_rot = x * np.cos(angle_radians) + y * np.sin(angle_radians)
    y_rot = -x * np.sin(angle_radians) + y * np.cos(angle_radians)

    x = x_rot * wavelength * np.pi / 180
    y = y_rot * wavelength * np.pi / 180

    moire = np.sin(x + y) * strength / 100
    moire = cv2.resize(moire, (width, height), interpolation=cv2.INTER_CUBIC)
    moire = np.repeat(moire[:, :, np.newaxis], source.shape[-1], axis=-1)
    moire = moire.astype(source.dtype)

    processed_image = cv2.addWeighted(
        source, 1, moire, strength / 100, 0).astype(np.uint8)

    return processed_image


def apply_black_rectangle(source: np.ndarray, random_args: bool = True, noise_type='gaussian', mean: float = 0, var: float = 0.001, rect_x: int = None, rect_y: int = None, rect_width: int = None, rect_height: int = None) -> np.ndarray:
    r"""
    Applies black rectangle to an image

    Parameters
    ----------
        - source : source image as `np.ndarray`

        - random_args : `bool`, default: `True`

            Randomizes arguments for this function call

            Overwrites all the other passed arguments

        - noise_type : `str` 
        
            Controls the noise type to be applied
            
            Either `'gaussian'` or `'s&p'`

        - mean : `float`, default: `0`

        - var : `float`, default: `0.001`

        - rect_x, rect_y : `int`

        - rect_width, rect_height : `int` 

    Returns
    -------
        - `np.ndarray` Processed image

    """
    if random_args:
        rect_x, rect_y, rect_width, rect_height, mean, var = random_arguments(
            [
                [int, 0, source.shape[1] // 2],
                [int, 0, source.shape[0] // 2],
                [int, source.shape[1] // 4, source.shape[1] // 2],
                [int, source.shape[0] // 4, source.shape[0] // 2],
                [float, -128, 128],
                [float, 0, 0.1]
            ]
        )
        noise_type = np.random.choice(['gaussian', 's&p'])

    rect_patch = np.zeros((rect_height, rect_width))
    rect_patch = rect_patch[:, :, np.newaxis]

    if noise_type == 'gaussian':
        rect_patch = random_noise(
            rect_patch, mode=noise_type, mean=mean, var=var)
    elif noise_type == 's&p':
        rect_patch = random_noise(rect_patch, mode='s&p')
    else:
        raise NotImplementedError(
            f"We currently don't support {noise_type} noise... Only 'gaussian' and 's&p'"
        )

    # Apply the patch to the source image
    processed_image = source.copy()
    processed_image[rect_y:rect_y+rect_height,
                    rect_x:rect_x+rect_width, :] = rect_patch

    return processed_image
