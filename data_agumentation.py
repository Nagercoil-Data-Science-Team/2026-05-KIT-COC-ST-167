# ============================================================
# DATA AUGMENTATION FOR PREPROCESSED DATASET
# ============================================================

import os
import cv2
import numpy as np

# ============================================================
# INPUT / OUTPUT FOLDERS
# ============================================================

# Preprocessed dataset
INPUT_FOLDER = "preprocessed_dataset"

# Augmented dataset
OUTPUT_FOLDER = "augmented_dataset"

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# ROTATION
# ------------------------------------------------------------

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )

    return rotated

# ------------------------------------------------------------
# FLIPPING
# ------------------------------------------------------------

def flip_image(image):

    flipped = cv2.flip(image, 1)

    return flipped

# ------------------------------------------------------------
# TRANSLATION
# ------------------------------------------------------------

def translate_image(image, x_shift, y_shift):

    h, w = image.shape[:2]

    matrix = np.float32([
        [1, 0, x_shift],
        [0, 1, y_shift]
    ])

    translated = cv2.warpAffine(
        image,
        matrix,
        (w, h)
    )

    return translated

# ------------------------------------------------------------
# SCALING
# ------------------------------------------------------------

def scale_image(image, scale_factor):

    h, w = image.shape[:2]

    scaled = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor
    )

    scaled = cv2.resize(
        scaled,
        (w, h)
    )

    return scaled

# ------------------------------------------------------------
# BRIGHTNESS ADJUSTMENT
# ------------------------------------------------------------

def adjust_brightness(image, value):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    v = np.clip(v + value, 0, 255)

    final_hsv = cv2.merge((h, s, v.astype(np.uint8)))

    bright = cv2.cvtColor(
        final_hsv,
        cv2.COLOR_HSV2BGR
    )

    return bright

# ============================================================
# PROCESS EACH CLASS
# ============================================================

classes = os.listdir(INPUT_FOLDER)

for class_name in classes:

    input_class_path = os.path.join(
        INPUT_FOLDER,
        class_name
    )

    if not os.path.isdir(input_class_path):
        continue

    # --------------------------------------------------------
    # CREATE OUTPUT CLASS FOLDER
    # --------------------------------------------------------

    output_class_path = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    os.makedirs(output_class_path, exist_ok=True)

    image_files = os.listdir(input_class_path)

    print(f"\nProcessing Class: {class_name}")

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for image_file in image_files:

        image_path = os.path.join(
            input_class_path,
            image_file
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        # ----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        original_save = os.path.join(
            output_class_path,
            "original_" + image_file
        )

        cv2.imwrite(original_save, image)

        # ----------------------------------------------------
        # ROTATION
        # ----------------------------------------------------

        rotated = rotate_image(image, 20)

        rotated_save = os.path.join(
            output_class_path,
            "rotated_" + image_file
        )

        cv2.imwrite(rotated_save, rotated)

        # ----------------------------------------------------
        # FLIPPING
        # ----------------------------------------------------

        flipped = flip_image(image)

        flipped_save = os.path.join(
            output_class_path,
            "flipped_" + image_file
        )

        cv2.imwrite(flipped_save, flipped)

        # ----------------------------------------------------
        # TRANSLATION
        # ----------------------------------------------------

        translated = translate_image(
            image,
            20,
            20
        )

        translated_save = os.path.join(
            output_class_path,
            "translated_" + image_file
        )

        cv2.imwrite(translated_save, translated)

        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        scaled = scale_image(
            image,
            1.2
        )

        scaled_save = os.path.join(
            output_class_path,
            "scaled_" + image_file
        )

        cv2.imwrite(scaled_save, scaled)

        # ----------------------------------------------------
        # BRIGHTNESS ADJUSTMENT
        # ----------------------------------------------------

        bright = adjust_brightness(
            image,
            40
        )

        bright_save = os.path.join(
            output_class_path,
            "bright_" + image_file
        )

        cv2.imwrite(bright_save, bright)

    print(f"Completed: {class_name}")

# ============================================================
# SAVE SAMPLE AUGMENTED IMAGES
# ============================================================

SAMPLE_FOLDER = "sample_augmentation_results"

os.makedirs(SAMPLE_FOLDER, exist_ok=True)

# ============================================================
# SAVE 3 SAMPLE IMAGES FROM EACH CLASS
# ============================================================

for class_name in classes:

    class_path = os.path.join(
        OUTPUT_FOLDER,
        class_name
    )

    if not os.path.exists(class_path):
        continue

    sample_images = os.listdir(class_path)[:3]

    for i, img_name in enumerate(sample_images):

        img_path = os.path.join(
            class_path,
            img_name
        )

        img = cv2.imread(img_path)

        if img is None:
            continue

        save_name = (
            class_name
            + "_sample_"
            + str(i+1)
            + ".png"
        )

        save_path = os.path.join(
            SAMPLE_FOLDER,
            save_name
        )

        cv2.imwrite(save_path, img)

# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("DATA AUGMENTATION COMPLETED")
print("AUGMENTED DATASET CREATED")
print("SAMPLE AUGMENTED IMAGES SAVED")
print("======================================")