import click
import cv2
import numpy as np
import pupil_apriltags as apriltags
import toml

from fiducial_server.camera import open_video_capture
from server.config import Config

WINDOW = "Window"


def find_h_detections(
    detections, homography_markers: tuple[int, int, int, int]
) -> dict[int, apriltags.Detection] | None:
    h_detections = {id: detections.get(id) for id in homography_markers}
    if all(h_detection is not None for h_detection in h_detections.values()):
        return h_detections
    return None


def draw_detections(frame, detections):
    for tag_id, detection in detections.items():
        if detection.corners is None or detection.center is None:
            continue

        corners = np.rint(np.asarray(detection.corners)).astype(np.int32)
        polygon = corners.reshape((-1, 1, 2))
        color = (0, 0, 255)

        w = 0
        cv2.fillConvexPoly(frame, polygon, (w, w, w), cv2.LINE_AA)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        center = tuple(np.rint(np.asarray(detection.center)).astype(np.int32))
        cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)

        if detection.pose_R is not None:
            rotation = np.asarray(detection.pose_R).reshape(3, 3)
            axis_length = max(20, int(np.linalg.norm(corners[1] - corners[0]) * 0.6))
            for axis_index, axis_color in ((0, (0, 255, 0)), (1, (255, 0, 0))):
                direction = rotation[:2, axis_index]
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 1e-6:
                    continue
                endpoint = (
                    int(center[0] + direction[0] / direction_norm * axis_length),
                    int(center[1] + direction[1] / direction_norm * axis_length),
                )
                cv2.arrowedLine(
                    frame, center, endpoint, axis_color, 2, cv2.LINE_AA, tipLength=0.2
                )

        label_x = int(corners[:, 0].min())
        label_y = max(20, int(corners[:, 1].min()) - 8)
        label = f"ID: {tag_id}"
        if detection.pose_t is not None:
            pose = np.asarray(detection.pose_t).reshape(-1)
            if pose.size >= 3:
                x, y, z = pose[:3]
                label += f" ({x:.2f}, {y:.2f}, {z:.2f} m)"

        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_crossbox(frame, color=(0, 255, 0), thickness=4):
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2

    x1 = int(cx - cy)
    x2 = int(cx + cy)
    y1 = 0
    y2 = h

    cv2.line(frame, (x1, y1), (x2, y1), color, thickness)
    cv2.line(frame, (x1, y2), (x2, y2), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y2), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y2), color, thickness)

    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.line(frame, (x2, y1), (x1, y2), color, thickness)


@click.command()
@click.option("--source", required=True)
@click.option("--config", required=True)
def main(source: str, config: str):
    config: Config = Config.from_dict(toml.load(config))
    capture, _ = open_video_capture(source)

    width, height = 1280, 800

    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_FOCUS, config.camera_focus)

    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    camera_params = (
        config.camera_focal_x,
        config.camera_focal_y,
        config.camera_center_x,
        config.camera_center_y,
    )

    detector: apriltags.Detector = apriltags.Detector(
        families=" ".join(config.tag_families),
        nthreads=2,
        quad_decimate=config.quad_decimate,
        quad_sigma=config.quad_sigma,
        refine_edges=config.refine_edges,
        decode_sharpening=config.decode_sharpening,
        debug=config.debug,
    )

    HOMOGRAPHY_MARKERS = (
        config.homography_top_left,
        config.homography_top_right,
        config.homography_bottom_left,
        config.homography_bottom_right,
    )

    h_matrix = np.eye(3)

    cx, cy = width / 2, height / 2
    x1 = cx - cy
    x2 = cx + cy
    y1 = 0
    y2 = height

    dst_points = np.float32(
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]  # pyright: ignore[reportArgumentType]
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # find detections
            detections: dict[int, apriltags.Detection] = {
                detection.tag_id: detection
                for detection in detector.detect(  # pyright: ignore[reportGeneralTypeIssues]
                    cv2.cvtColor(
                        frame, cv2.COLOR_BGR2GRAY
                    ),  # pyright: ignore[reportArgumentType]
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=config.tag_size,
                )
            }

            if h_detections := find_h_detections(detections, HOMOGRAPHY_MARKERS):
                src_points = np.float32(
                    [  # pyright: ignore[reportArgumentType]
                        detection.center for detection in h_detections.values()
                    ]
                )
                h_matrix = cv2.getPerspectiveTransform(
                    src_points, dst_points
                )  # pyright: ignore[reportArgumentType, reportCallIssue]

            for detection in detections.values():
                if detection.center is None or detection.corners is None:
                    continue

                center = np.asarray(detection.center, dtype=np.float32).reshape(1, 1, 2)
                corners = np.asarray(detection.corners, dtype=np.float32).reshape(
                    -1, 1, 2
                )
                detection.center = cv2.perspectiveTransform(center, h_matrix).reshape(
                    2
                )  # pyright: ignore[reportAttributeAccessIssue]
                detection.corners = cv2.perspectiveTransform(corners, h_matrix).reshape(
                    -1, 2
                )  # pyright: ignore[reportAttributeAccessIssue]

            view = np.zeros((height, width, 3), dtype=np.uint8)
            draw_crossbox(view, color=(0, 64, 0), thickness=2)
            draw_detections(view, detections)

            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
