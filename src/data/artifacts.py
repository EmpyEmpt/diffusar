import numpy as np
import cv2


def apply_ghosting(source: np.ndarray, trail_length: int = 10, trail_decay: float = 0.8, shift_x: int = 5, shift_y: int = 5) -> np.ndarray:
    """
    Applies ghosting artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    trail_length: int, default: 10
        Controls the amount of 'ghost' images 

    trail_decay: float, default: 0.8
        Controls the decay of 'ghost' images

    shift_x, shift_y: int, default: 5
        Controls the shift of 'ghost' images from original position in pixel term

    Returns
    -------
    processed_image: np.ndarray
    """
    height, width, channels = source.shape
    ghosted_img = np.zeros((height, width, channels), np.float32)

    for i in range(trail_length):
        shift = np.array([shift_x * i, shift_y * i], np.float32)
        shifted = cv2.warpAffine(
            source,
            np.float32(
                [[1, 0, shift[0]],
                 [0, 1, shift[1]]]
            ),
            (width, height)
        )
        ghosted_img += shifted * trail_decay**i

    ghosted_img = ghosted_img / np.max(ghosted_img) * 255

    processed_image = np.uint8(np.clip(ghosted_img, 0, 255))
    return processed_image


def apply_distortion(source: np.ndarray, displacement: int = 10) -> np.ndarray:
    """
    Applies distortion artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    displacement: int, default: 10

    Returns
    -------
    processed_image: np.ndarray

    """
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


def apply_color_banding(source: np.ndarray, levels: int = 8) -> np.ndarray:
    """
    Applies color banding artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    levels: int, default: 8

    Returns
    -------
    processed_image: np.ndarray

    """
    # height, width, channels = img.shape
    processed_image = np.uint8(
        np.round(source / (256 / levels))
    ) * (256 // levels)

    return processed_image


def apply_silk_screen_effect(source: np.ndarray, levels: int = 8, threshold: int = 128) -> np.ndarray:
    """
    Applies silk screen artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    levels: int, default: 8

    threshold: int, default: 128

    Returns
    -------
    processed_image: np.ndarray

    """
    # height, width, channels = img.shape
    quantized = np.uint8(np.round(source / (256 / levels))) * (256 // levels)
    bw = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
    processed_image = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return processed_image


def apply_rainbow_effect(source: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Applies rainbow artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    strength: int, default: 10

    Returns
    -------
    processed_image: np.ndarray

    """
    height, width, channels = source.shape

    hue_map = np.zeros((height, width, 3), dtype=np.float32)
    hue_map[:, :, 0] = np.linspace(0, strength, width)
    hue_map = cv2.GaussianBlur(hue_map, (5, 5), 0)
    hue_map = hue_map * 360 / strength

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] + hue_map[:, :, 0]
    hsv[:, :, 0][hsv[:, :, 0] > 360] = hsv[:, :, 0][hsv[:, :, 0] > 360] - 360

    processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return processed_image


def apply_noise(source: np.ndarray, noise_type='gaussian', mean: float = 0, var: float = 0.001) -> np.ndarray:
    """
    Applies rainbow artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    noise_type: str, default: 'gaussian'
        Controls type of noise to be applied

    mean: float, default: 0

    var: float, default: 0.001

    Returns
    -------
    processed_image: np.ndarray

    """
    processed_image = source.copy()
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, var ** 0.5, source.shape)
        processed_image = np.clip(source + noise, 0, 255).astype(np.uint8)

    return processed_image


def apply_screen_tearing(source: np.ndarray, strength: float = 0.5, direction='horizontal') -> np.ndarray:
    """
    Applies rainbow artifact to an image

    Parameters
    ----------
    source : source image as ndarray

    direction: str, default: 'gaussian'
        Controls the direction of the tier
        Can be either 'horizontal' or 'vertical'

    strength: float, default: 0.5

    Returns
    -------
    processed_image: np.ndarray

    """

    height, width, channels = source.shape
    processed_image = np.zeros(source.shape, dtype=np.uint8)

    if direction == 'horizontal':
        tearing_height = int(strength * height)
        processed_image[:tearing_height, :, :] = source[:tearing_height, :, :]
        processed_image[tearing_height:, :, :] = source[:-tearing_height, :, :]
    elif direction == 'vertical':
        tearing_width = int(strength * width)
        processed_image[:, :tearing_width, :] = source[:, :tearing_width, :]
        processed_image[:, tearing_width:, :] = source[:, :-tearing_width, :]
    else:
        raise ValueError("Invalid direction")

    return processed_image


def apply_compression_artifact(source: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Applies jpeg compression to an image

    Parameters
    ----------
    source : source image as ndarray

    quality: int, default: 50

    Returns
    -------
    processed_image: np.ndarray

    """

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', source, params)
    processed_image = cv2.imdecode(enc, 1)

    return processed_image


def apply_moire_pattern(source: np.ndarray, strength: int = 50, wavelength: int = 100, angle: int = 45, grid_size: int = 10) -> np.ndarray:
    """
    Applies moire pattern to an image

    Parameters
    ----------
    source : source image as ndarray

    strength: int, default: 50

    wavelength: int, default: 100

    angle: int, default: 45

    grid_size: int, default: 10

    Returns
    -------
    processed_image: np.ndarray

    """
    height, width = source.shape[:2]

    x, y = np.meshgrid(np.linspace(0, width, grid_size),
                       np.linspace(0, height, grid_size))
    x = x * wavelength * np.pi / 180
    y = y * wavelength * np.pi / 180
    moire = np.sin(x + y) * strength / 100
    moire = cv2.resize(moire, (width, height), interpolation=cv2.INTER_CUBIC)
    moire = np.repeat(moire[:, :, np.newaxis], source.shape[-1], axis=-1)
    moire = moire.astype(source.dtype)

    processed_image = cv2.addWeighted(
        source, 1, moire, strength / 100, 0).astype(np.uint8)

    return processed_image
