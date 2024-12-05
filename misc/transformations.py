from io import BytesIO
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# Hilfsfunktion zum Auswählen eines zufälligen Rechtecks im Bild
def get_random_bbox(img:Image, crop_size=(50, 50), crop_pos=None):
    """Returns a random rectangle in the image.
        Args:
        img (PIL.Image): The input image.
        crop_size (tuple): The size of the rectangle to be cropped.
        crop_pos (str): The position of the rectangle to be cropped.
            If None, the rectangle will be cropped randomly.
            If 'center', the rectangle will be cropped from the center of the image.
    
    """
    width, height = img.size
    crop_width, crop_height = crop_size

    if crop_pos == 'center':
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

    elif crop_pos == 'top-left':
        left = 0
        top = 0
        right = crop_width
        bottom = crop_height

    elif crop_pos == 'top-right':
        left = width - crop_width
        top = 0
        right = width
        bottom = crop_height

    elif crop_pos == 'bottom-left':
        left = 0
        top = height - crop_height
        right = crop_width
        bottom = height

    elif crop_pos == 'bottom-right':
        left = width - crop_width
        top = height - crop_height
        right = width
        bottom = height

    else:
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

    
    return (left, top, right, bottom)

# Transformationen für Teilbereiche definieren
def apply_occlusion_patch(img, crop_size=(50, 50), crop_pos=None):
    """crop a part of the image and replace it with a uniform color."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    img = img.convert("RGB")  # Ensures the image is in RGB mode

    # black
    img.paste((0, 0, 0), bbox)

    return img

def apply_grayscale_patch(img, crop_size=(50, 50), crop_pos=None):
    """Randomly convert a part of the image in grayscale."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    patch = ImageOps.grayscale(patch).convert("RGB")  # In Graustufen umwandeln
    img.paste(patch, bbox)
    return img

def apply_blur_patch(img, crop_size=(50, 50), blur_radius=5, crop_pos=None):
    """Apply a blur effect to a part of the image based on crop position."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    patch = patch.filter(ImageFilter.GaussianBlur(blur_radius))  # Unschärfe anwenden
    img.paste(patch, bbox)
    return img

def apply_noise_patch(img, crop_size=(50, 50), crop_pos=None):
    """Apply noise to a part of the image based on crop position."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    # Rauschen hinzufügen
    noise = np.random.normal(0, 25, patch.size + (3,))  # Rauschen mit mittlerer Intensität
    patch = np.array(patch) + noise
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    patch = Image.fromarray(patch)
    img.paste(patch, bbox)
    return img

def apply_brightness_patch(img, crop_size=(50, 50), brightness_factor=2, crop_pos=None):
    """Adjust the brightness of a part of the image based on crop position."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    enhancer = ImageEnhance.Brightness(patch)
    patch = enhancer.enhance(brightness_factor)  # Helligkeit anpassen
    img.paste(patch, bbox)
    return img

def apply_contrast_patch(img, crop_size=(50, 50), contrast_factor=2, crop_pos=None):
    """Adjust the contrast of a part of the image based on crop position."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    enhancer = ImageEnhance.Contrast(patch)
    patch = enhancer.enhance(contrast_factor)  # Kontrast anpassen
    img.paste(patch, bbox)
    return img

def apply_compression_patch(img, crop_size=(50, 50), quality=25, crop_pos=None):
    """Compress a part of the image based on crop position."""
    img = img.copy()
    bbox = get_random_bbox(img, crop_size, crop_pos=crop_pos)
    patch = img.crop(bbox)
    # Bild in Bytes umwandeln
    buffer = BytesIO()
    patch.save(buffer, format="JPEG", quality=quality)
    # Komprimiertes Bild zurück in ein Bildobjekt laden
    patch = Image.open(buffer)
    img.paste(patch, bbox)
    return img