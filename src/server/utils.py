import pupil_apriltags as apriltags
import scipy

type Marker = tuple[float, float, float, float, float, float]


def marker_from_detection(detection: apriltags.Detection) -> Marker | None:
    try:
        x, y, z = detection.pose_t.flatten()  # pyright: ignore[reportAttributeAccessIssue]
        euler_x, euler_y, euler_z = scipy.spatial.transform.Rotation.from_matrix(
            detection.pose_R
        ).as_euler("xyz", degrees=True)

        return (x, y, z, euler_x, euler_y, euler_z)
    except Exception:  # noqa: BLE001
        return None

def make_address(parts):
    return (parts[0], int(parts[1]))
