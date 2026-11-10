import click
import cv2

from fiducial_server.camera import open_video_capture

WINDOW = "Camera View"


@click.command()
@click.option(
    "--source", default="0", show_default=True, help="Camera index or video source."
)
def main(source: str) -> None:
    capture, (camera_width, camera_height) = open_video_capture(source)
    click.echo(f"Camera resolution: {camera_width}x{camera_height}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
