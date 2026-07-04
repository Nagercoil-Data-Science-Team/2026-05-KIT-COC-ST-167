# ============================================================
# STAGE 8 — SPATIOTEMPORAL ALIGNMENT
# ============================================================

import os
import cv2
import numpy as np

# ============================================================
# INPUT / OUTPUT FOLDERS
# ============================================================

INPUT_FOLDER = "temporal_sequences"

OUTPUT_FOLDER = "aligned_sequences"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# FEATURE ALIGNMENT FUNCTION
# ============================================================

def feature_alignment(reference_frame, target_frame):

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    ref_gray = cv2.cvtColor(
        reference_frame,
        cv2.COLOR_BGR2GRAY
    )

    tar_gray = cv2.cvtColor(
        target_frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # ORB Feature Detector
    # --------------------------------------------------------

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(ref_gray, None)

    kp2, des2 = orb.detectAndCompute(tar_gray, None)

    # --------------------------------------------------------
    # If descriptors missing
    # --------------------------------------------------------

    if des1 is None or des2 is None:

        return target_frame

    # --------------------------------------------------------
    # Feature Matching
    # --------------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(des1, des2)

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    # --------------------------------------------------------
    # Minimum matches check
    # --------------------------------------------------------

    if len(matches) < 10:

        return target_frame

    # --------------------------------------------------------
    # Extract matched points
    # --------------------------------------------------------

    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    # --------------------------------------------------------
    # Homography Matrix
    # --------------------------------------------------------

    H, mask = cv2.findHomography(
        dst_pts,
        src_pts,
        cv2.RANSAC,
        5.0
    )

    # --------------------------------------------------------
    # If homography fails
    # --------------------------------------------------------

    if H is None:

        return target_frame

    # --------------------------------------------------------
    # Warp target image
    # --------------------------------------------------------

    h, w = reference_frame.shape[:2]

    aligned = cv2.warpPerspective(
        target_frame,
        H,
        (w, h)
    )

    return aligned

# ============================================================
# OPTICAL FLOW ALIGNMENT
# ============================================================

def optical_flow_alignment(reference_frame, target_frame):

    ref_gray = cv2.cvtColor(
        reference_frame,
        cv2.COLOR_BGR2GRAY
    )

    tar_gray = cv2.cvtColor(
        target_frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Dense Optical Flow
    # --------------------------------------------------------

    flow = cv2.calcOpticalFlowFarneback(
        ref_gray,
        tar_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # --------------------------------------------------------
    # Flow magnitude and angle
    # --------------------------------------------------------

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(reference_frame)

    hsv[..., 1] = 255

    hsv[..., 0] = angle * 180 / np.pi / 2

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    optical_flow_visual = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return optical_flow_visual

# ============================================================
# PROCESS ALL SEQUENCES
# ============================================================

sequence_folders = os.listdir(INPUT_FOLDER)

for sequence_name in sequence_folders:

    sequence_path = os.path.join(
        INPUT_FOLDER,
        sequence_name
    )

    if not os.path.isdir(sequence_path):
        continue

    print(f"\nProcessing Sequence: {sequence_name}")

    # --------------------------------------------------------
    # CREATE OUTPUT FOLDER
    # --------------------------------------------------------

    output_sequence_path = os.path.join(
        OUTPUT_FOLDER,
        sequence_name
    )

    os.makedirs(output_sequence_path, exist_ok=True)

    # --------------------------------------------------------
    # Read frames in order
    # --------------------------------------------------------

    frame_files = sorted(
        os.listdir(sequence_path)
    )

    frames = []

    for frame_file in frame_files:

        frame_path = os.path.join(
            sequence_path,
            frame_file
        )

        frame = cv2.imread(frame_path)

        if frame is not None:
            frames.append((frame_file, frame))

    # --------------------------------------------------------
    # Need minimum 2 frames
    # --------------------------------------------------------

    if len(frames) < 2:
        continue

    # ========================================================
    # REFERENCE FRAME
    # ========================================================

    reference_name, reference_frame = frames[0]

    # Save original reference frame
    cv2.imwrite(
        os.path.join(
            output_sequence_path,
            reference_name
        ),
        reference_frame
    )

    # ========================================================
    # ALIGN OTHER FRAMES
    # ========================================================

    for i in range(1, len(frames)):

        current_name, current_frame = frames[i]

        # ----------------------------------------------------
        # FEATURE ALIGNMENT
        # ----------------------------------------------------

        aligned_frame = feature_alignment(
            reference_frame,
            current_frame
        )

        # ----------------------------------------------------
        # SAVE ALIGNED FRAME
        # ----------------------------------------------------

        aligned_save = os.path.join(
            output_sequence_path,
            "aligned_" + current_name
        )

        cv2.imwrite(
            aligned_save,
            aligned_frame
        )

        # ----------------------------------------------------
        # OPTICAL FLOW VISUALIZATION
        # ----------------------------------------------------

        optical_flow = optical_flow_alignment(
            reference_frame,
            aligned_frame
        )

        optical_save = os.path.join(
            output_sequence_path,
            "opticalflow_" + current_name
        )

        cv2.imwrite(
            optical_save,
            optical_flow
        )

    print(f"Aligned: {sequence_name}")

# ============================================================
# SAMPLE OUTPUTS
# ============================================================

SAMPLE_FOLDER = "sample_alignment_results"

os.makedirs(SAMPLE_FOLDER, exist_ok=True)

sample_sequences = os.listdir(OUTPUT_FOLDER)[:3]

for seq in sample_sequences:

    seq_path = os.path.join(
        OUTPUT_FOLDER,
        seq
    )

    if not os.path.isdir(seq_path):
        continue

    frame_files = os.listdir(seq_path)

    for frame_file in frame_files:

        frame_path = os.path.join(
            seq_path,
            frame_file
        )

        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        save_name = seq + "_" + frame_file

        save_path = os.path.join(
            SAMPLE_FOLDER,
            save_name
        )

        cv2.imwrite(save_path, frame)

# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("SPATIOTEMPORAL ALIGNMENT COMPLETED")
print("SPATIAL ALIGNMENT DONE")
print("TEMPORAL ALIGNMENT DONE")
print("OPTICAL FLOW GENERATED")
print("ALIGNED SEQUENCES SAVED")
print("======================================")