import sys
from pathlib import Path

import click
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from fiducial_server.camera import open_video_capture


def print_reprojection_graph(errors: list[float]) -> None:
    console = Console()
    table = Table(title="Per-view reprojection error")

    table.add_column("View", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Graph", min_width=30)

    # Keep the scale consistent: bars reach full width at 2 pixels.
    scale_max = 2.0
    bar_width = 30

    for index, error in enumerate(errors, start=1):
        if error < 0.5:
            color, label = "green", "GOOD"
        elif error < 1.0:
            color, label = "yellow", "MEH"
        else:
            color, label = "red", "BAD"

        filled = round(min(error / scale_max, 1.0) * bar_width)

        bar = Text("█" * filled, style=color)
        bar.append("░" * (bar_width - filled), style="dim")

        table.add_row(
            str(index),
            f"{error:.3f} px ({label})",
            bar,
        )

    console.print()
    console.print(table)
    console.print(
        "[green]GOOD[/green] < 0.5 px  "
        "[yellow]MEH[/yellow] 0.5–1.0 px  "
        "[red]BAD[/red] ≥ 1.0 px"
    )


def capture_samples(capture, detector, min_corners, board_points, min_samples: int):
    captured_samples = 0

    try:
        while captured_samples < min_samples:
            ok, frame = capture.read()

            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
                count = 0 if ids is None else len(ids)

                if any([corners is None, ids is None, count < min_corners]):
                    continue

                xy = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
                idx = np.asarray(ids, dtype=np.int32).reshape(-1)

                yield board_points[idx].reshape(-1, 1, 3).copy(), xy.reshape(
                    -1, 1, 2
                ).copy()
                captured_samples += 1

            except cv2.error as exception:
                print("detector.detect() threw exception:", exception)
    finally:
        capture.release()


@click.command()
@click.option("--source", required=True)
@click.option("--squares", nargs=2, type=int)
@click.option("--square_size", type=float)
@click.option("--marker_size", type=float)
@click.option("--min_corners", type=int, default=8)
@click.option("--dictionary", type=str)
@click.option("--output", type=Path)
def main(
    source: str, squares, square_size, marker_size, min_corners, dictionary, output
):
    capture, image_size = open_video_capture(source)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))

    board = cv2.aruco.CharucoBoard(squares, square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points, image_points = [], []

    n_samples = 100

    for board_point, image_point in tqdm(
        capture_samples(capture, detector, min_corners, board_points, n_samples)
    ):
        object_points.append(board_point)
        image_points.append(image_point)

    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )

        per_view = []
        for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            per_view.append(
                float(
                    cv2.norm(img, projected, cv2.NORM_L2)
                    / np.sqrt(max(len(projected), 1))
                )
            )

        print_reprojection_graph(per_view)

    except cv2.error as exception:
        print("cv2.calibrateCamera() threw exception:", exception)
        return SystemExit(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(output), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        print(f"Could not write {output}", file=sys.stderr)
        return SystemExit(2)

    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_reprojection_error_px", float(rms))
    fs.write("views_used", len(image_points))
    fs.release()


if __name__ == "__main__":
    main()
