from pathlib import Path
from datetime import datetime
import csv
import os

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR

os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig")
)

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIXELS_PER_MM = 10.0
K = 200.0
COMPRESSION_MM = 5.0
TRUE_FORCE_N = K * COMPRESSION_MM / 1000.0

ANGLES = [0, 5, 10, 20, 30, 45, 60]
METHODS = ["Row occupancy", "Rotated bounding box"]


def segment(image):
    """Identify tissue pixels using HSV thresholds."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv,
        np.array([0, 35, 40], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([170, 35, 40], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    return cv2.bitwise_or(mask1, mask2)


def row_height(mask):
    counts = np.count_nonzero(mask, axis=1)
    peak = counts.max()

    if peak == 0:
        return None

    valid = counts >= 0.5 * peak
    padded = np.concatenate(([False], valid, [False]))
    changes = np.diff(padded.astype(np.int8))

    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    if starts.size == 0:
        return None

    return float((ends - starts).max())


def rotated_height(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None, None

    rect = cv2.minAreaRect(largest)
    width, height = rect[1]
    short_side = min(width, height)

    return float(short_side), cv2.boxPoints(rect)


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot save: {path}")


def main():
    reference = np.full((480,640,3),245,dtype=np.uint8)
    cv2.rectangle(reference,(170,150),(470,350),(150,150,230),thickness=-1)

    reference_mask = segment(reference)
    short_side, _ = rotated_height(reference_mask)

    reference_heights = {
        "Row occupancy": row_height(reference_mask),
        "Rotated bounding box": short_side,
    }

    if any(v is None for v in reference_heights.values()):
        raise RuntimeError("Reference detection failed")

    # Generate fixed compression; rotation does not change force.
    compressed = np.full_like(reference, 245)
    top = 150 + round(COMPRESSION_MM * PIXELS_PER_MM)
    bottom = 350

    cv2.rectangle(
        compressed,
        (170, top),
        (470, bottom),
        (150, 150, 230),
        thickness=-1,
    )

    center = (320.0, (top + bottom) / 2.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = BASE_DIR / f"day4_rotation_compare_{timestamp}"
    output_dir.mkdir(exist_ok=False)

    rows = []
    estimates = {method: [] for method in METHODS}
    errors = {method: [] for method in METHODS}

    fig_samples, axes = plt.subplots(
        2, 4, figsize=(12, 7), constrained_layout=True
    )

    print(f"Ground truth: {TRUE_FORCE_N:.3f} N")
    print(f"Reference measurements: {reference_heights}")

    for ax, angle in zip(axes.flat, ANGLES):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            compressed,
            matrix,
            (640, 480),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(245, 245, 245),
        )

        mask = segment(rotated)
        short_side, box = rotated_height(mask)

        heights = {
            "Row occupancy": row_height(mask),
            "Rotated bounding box": short_side,
        }

        for method in METHODS:
            height = heights[method]
            force = None
            signed_error = None

            if height is not None:
                compression_mm = (
                    reference_heights[method] - height
                ) / PIXELS_PER_MM
                force = K * compression_mm / 1000.0
                signed_error = force - TRUE_FORCE_N

                print(
                    f"{angle:>2} deg | {method:<21} | "
                    f"Force={force:.4f} N | "
                    f"Error={signed_error:+.4f} N"
                )
            else:
                print(f"{angle:>2} deg | {method}: failed")

            rows.append([
                angle,
                method,
                height,
                TRUE_FORCE_N,
                force,
                signed_error,
                abs(signed_error)
                if signed_error is not None else None,
            ])

            estimates[method].append(
                force if force is not None else np.nan
            )
            errors[method].append(
                signed_error
                if signed_error is not None else np.nan
            )

        # Draw the fitted rotated box for visual inspection.
        annotated = rotated.copy()
        if box is not None:
            cv2.polylines(
                annotated,
                [np.rint(box).astype(np.int32)],
                True,
                (0, 160, 0),
                2,
            )

        save_image(
            output_dir / f"rotation_{angle:02d}.png",
            annotated,
        )
        save_image(
            output_dir / f"rotation_{angle:02d}_mask.png",
            mask,
        )

        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{angle} degrees")
        ax.axis("off")

    for ax in list(axes.flat)[len(ANGLES):]:
        ax.axis("off")

    fig_samples.suptitle(
        "Fixed compression: 5 mm | Green: rotated bounding box"
    )
    fig_samples.savefig(
        output_dir / "rotation_samples.png", dpi=160
    )
    plt.close(fig_samples)

    with (output_dir / "measurements.csv").open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "angle_deg",
            "method",
            "measured_height_px",
            "true_force_n",
            "estimated_force_n",
            "signed_error_n",
            "absolute_error_n",
        ])
        writer.writerows(rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )

    for method in METHODS:
        axes[0].plot(
            ANGLES, estimates[method],
            marker="o", label=method,
        )
        axes[1].plot(
            ANGLES, errors[method],
            marker="o", label=method,
        )

    axes[0].axhline(
        TRUE_FORCE_N,
        color="black", linestyle="--",
        label="Ground truth",
    )
    axes[0].set(
        xlabel="Rotation angle (degrees)",
        ylabel="Estimated force (N)",
        title="Force estimation",
    )

    axes[1].axhline(
        0, color="black", linestyle="--"
    )
    axes[1].set(
        xlabel="Rotation angle (degrees)",
        ylabel="Signed error (N)",
        title="Error at each angle",
    )

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.savefig(
        output_dir / "method_comparison.png", dpi=160
    )
    plt.close(fig)

    print("\nMAE across tested angles (detected samples only):")
    for method in METHODS:
        values = np.array(errors[method])
        valid = values[np.isfinite(values)]
        if valid.size:
            print(
                f"{method}: {valid.size}/{len(ANGLES)} detected, "
                f"MAE={np.mean(np.abs(valid)):.4f} N"
            )
        else:
            print(f"{method}: detection failed")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()