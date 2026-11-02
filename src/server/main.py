import socket
from pathlib import Path

import click
import cv2
import numpy as np
import pupil_apriltags as apriltags

from server.config import Config
from server.utils import Marker, marker_from_detection


@click.command()
@click.option("--headless", is_flag=True, default=False)
@click.option("--source", required=True)
def main(headless: bool, source: str) -> None:
    config = Config.from_file(Path("./src/fiducial_server/config.toml"))
    config.source = int(source) if source.isdigit() else source

    capture = cv2.VideoCapture(config.source)
    assert capture.isOpened(), "Failed to open camera"
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            detections: dict[int, apriltags.Detection] = {
                detection.tag_id: detection
                for detection in detector.detect(  # pyright: ignore[reportGeneralTypeIssues]
                    frame,  # pyright: ignore[reportArgumentType]
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=config.tag_size
                )
            }

            markers: dict[int, Marker] = {
                id: marker
                for id, detection in detections.items()
                if (marker := marker_from_detection(detection)) is not None
            }

            # TODO: normalize coordinate systems through homography markers

            if len(markers) > 0:
                message = ';'.join(f"{id}:{','.join(f"{value:.{digits}f}" for value in marker)}" for id, marker in markers.items())
                sock.sendto(message.encode("utf-8"), config.udp)
    finally:
        sock.close()
        capture.release()

if __name__ == "__main__":
    main()
