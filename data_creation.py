# ============================================================
# SUBSTATION EQUIPMENT CROPPING FROM JSON ANNOTATION
# ============================================================

import os
import cv2
import json
import numpy as np

# ============================================================
# INPUT / OUTPUT FOLDERS
# ============================================================

# Main dataset folder
BASE_FOLDER = os.getcwd()

# Image folder
IMAGE_FOLDER = os.path.join(BASE_FOLDER, "images")

# Annotation folder
ANNOTATION_FOLDER = os.path.join(BASE_FOLDER, "annotation")

# Output folder
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "cropped_dataset")

# ============================================================
# CHECK INPUT FOLDERS
# ============================================================

if not os.path.exists(IMAGE_FOLDER):
    print(f"Image folder not found:\n{IMAGE_FOLDER}")
    exit()

if not os.path.exists(ANNOTATION_FOLDER):
    print(f"Annotation folder not found:\n{ANNOTATION_FOLDER}")
    exit()

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\n======================================")
print("INPUT IMAGE FOLDER:")
print(IMAGE_FOLDER)

print("\nINPUT ANNOTATION FOLDER:")
print(ANNOTATION_FOLDER)

print("\nOUTPUT FOLDER:")
print(OUTPUT_FOLDER)
print("======================================\n")

# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = []

for file in os.listdir(IMAGE_FOLDER):

    ext = os.path.splitext(file)[1].lower()

    if ext in valid_extensions:
        image_files.append(file)

# ============================================================
# CHECK IMAGE COUNT
# ============================================================

print(f"Total Images Found: {len(image_files)}\n")

# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for image_file in image_files:

    print(f"Processing: {image_file}")

    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_path = os.path.join(IMAGE_FOLDER, image_file)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Cannot read image: {image_file}")
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # JSON FILE NAME
    # --------------------------------------------------------

    json_name = os.path.splitext(image_file)[0] + ".json"

    json_path = os.path.join(ANNOTATION_FOLDER, json_name)

    # --------------------------------------------------------
    # CHECK JSON FILE
    # --------------------------------------------------------

    if not os.path.exists(json_path):
        print(f"JSON annotation missing: {json_name}")
        continue

    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    try:

        with open(json_path, "r") as f:
            annotation_data = json.load(f)

    except Exception as e:

        print(f"JSON Error in {json_name}")
        print(e)
        continue

    # --------------------------------------------------------
    # CHECK OBJECTS
    # --------------------------------------------------------

    if "objects" not in annotation_data:
        print(f"No objects found in: {json_name}")
        continue

    objects = annotation_data["objects"]

    object_count = 0

    # ========================================================
    # PROCESS EACH OBJECT
    # ========================================================

    for obj in objects:

        try:

            # ------------------------------------------------
            # CLASS NAME
            # ------------------------------------------------

            class_name = obj["classTitle"]

            # ------------------------------------------------
            # CREATE CLASS OUTPUT FOLDER
            # ------------------------------------------------

            class_folder = os.path.join(
                OUTPUT_FOLDER,
                class_name
            )

            os.makedirs(class_folder, exist_ok=True)

            # ------------------------------------------------
            # POLYGON COORDINATES
            # ------------------------------------------------

            polygon = obj["points"]["exterior"]

            polygon_points = np.array(
                polygon,
                dtype=np.int32
            )

            # ------------------------------------------------
            # CREATE MASK
            # ------------------------------------------------

            mask = np.zeros(
                (height, width),
                dtype=np.uint8
            )

            # ------------------------------------------------
            # DRAW POLYGON
            # ------------------------------------------------

            cv2.fillPoly(
                mask,
                [polygon_points],
                255
            )

            # ------------------------------------------------
            # APPLY MASK
            # ------------------------------------------------

            segmented = cv2.bitwise_and(
                image,
                image,
                mask=mask
            )

            # ------------------------------------------------
            # GET BOUNDING BOX
            # ------------------------------------------------

            x, y, w, h = cv2.boundingRect(
                polygon_points
            )

            cropped_object = segmented[
                y:y+h,
                x:x+w
            ]

            # ------------------------------------------------
            # CHECK EMPTY IMAGE
            # ------------------------------------------------

            if cropped_object.size == 0:
                continue

            # ------------------------------------------------
            # SAVE IMAGE NAME
            # ------------------------------------------------

            save_name = (
                os.path.splitext(image_file)[0]
                + "_"
                + str(object_count)
                + ".png"
            )

            save_path = os.path.join(
                class_folder,
                save_name
            )

            # ------------------------------------------------
            # SAVE CROPPED IMAGE
            # ------------------------------------------------

            cv2.imwrite(
                save_path,
                cropped_object
            )

            object_count += 1

        except Exception as e:

            print(f"Object processing error in {image_file}")
            print(e)

    print(f"Saved Objects: {object_count}\n")

# ============================================================
# FINISHED
# ============================================================

print("======================================")
print("DATASET PROCESSING COMPLETED")
print("Cropped images saved successfully")
print("======================================")