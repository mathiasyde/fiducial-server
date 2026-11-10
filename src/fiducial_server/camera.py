import cv2


class CameraException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def open_video_capture(
    source: str, size: tuple[int, int] = (1920, 1080)
) -> tuple[cv2.VideoCapture, tuple[int, int]]:
    capture = cv2.VideoCapture(
        int(source) if source.isdigit() else source, cv2.CAP_V4L2
    )
    width, height = size

    capture.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
    )  # pyright: ignore[reportAttributeAccessIssue]
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        capture.release()
        raise CameraException("Failed to open camera")

    capture_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return capture, (capture_width, capture_height)
