import sys
from pathlib import Path

import click
import cv2
import numpy as np


from rich.console import Console
from rich.table import Table
from rich.text import Text


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



@click.command()
@click.option("--source", type=str)
@click.option("--squares", nargs=2, type=int)
@click.option("--square_size", type=float)
@click.option("--marker_size", type=float)
@click.option("--min_corners", type=int, default=8)
@click.option("--dictionary", type=str)
@click.option("--output", type=Path)
@click.option("--debug", type=bool, is_flag=True, default=False)
def main(source, squares, square_size, marker_size, min_corners, dictionary, output, debug):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))

    board = cv2.aruco.CharucoBoard(squares, square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    object_points, image_points = [], []
    image_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

                count = 0 if ids is None else len(ids)

                if debug:
                    view = frame.copy()

                    if marker_ids is not None and len(marker_ids):
                        cv2.aruco.drawDetectedMarkers(view, marker_corners, marker_ids)

                    if corners is not None and ids is not None and count:
                        c = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
                        i = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
                        if len(c) == len(i):
                            cv2.aruco.drawDetectedCornersCharuco(view, c, i)

                    markers = 0 if marker_ids is None else len(marker_ids)
                    cv2.putText(view, f"Markers: {markers}  ChArUco corners: {count}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 0) if count >= min_corners else (0, 165, 255),
                                2, cv2.LINE_AA)
                    cv2.imshow("ChArUco camera calibration", view)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break

                if any([corners is None, ids is None, count < min_corners]):
                    continue

                xy = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
                idx = np.asarray(ids, dtype=np.int32).reshape(-1)
                object_points.append(board_points[idx].reshape(-1, 1, 3).copy())
                image_points.append(xy.reshape(-1, 1, 2).copy())

                # TODO: proper termination logic
                if len(object_points) > 20:
                    break
            except cv2.error as exception:
                print("detector.detect() threw exception:", exception)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if len(image_points) < 8:
        return SystemExit(2)

    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )

        if debug:
            per_view = []
            for obj, img, rv, tv in zip(object_points, image_points, rvecs, tvecs):
                    projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
                    per_view.append(
                        float(cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(max(len(projected), 1)))
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
