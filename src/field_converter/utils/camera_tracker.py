from dataclasses import dataclass, field
from collections import deque
import numpy as np
import cv2
import scipy.optimize
from typing import Tuple, Optional


class Debugger:
    """
    Centralized visualization manager for debugging camera tracking.
    """

    def __init__(self, debug_stages: Tuple[str, ...] = ("projection",)):
        """
        Args:
            enabled: Master switch for all visualizations
            stages: stages to visualize, e.g., ('flow', 'mask')
        """
        self.stages = set(debug_stages)
        self.frame_curr = None

    def update(self, frame):
        self.frame_curr = frame

    @property
    def visualize(self) -> bool:
        return len(self.stages) > 0

    def draw_optical_flow(
        self,
        pts_prev: np.ndarray,
        pts_next: np.ndarray,
        status: np.ndarray,
    ) -> None:
        """Visualize optical flow vectors."""
        if "flow" not in self.stages:
            return
        vis = self.frame_curr
        for i in range(len(pts_prev)):
            if not status[i]:
                continue
            pt1 = tuple(pts_prev[i].astype(int))
            pt2 = tuple(pts_next[i].astype(int))
            cv2.circle(vis, pt1, 5, (0, 0, 255), -1)  # Red: previous
            cv2.circle(vis, pt2, 5, (0, 255, 0), -1)  # Green: current
            cv2.line(vis, pt1, pt2, (0, 255, 255), 1)  # Yellow: flow vector

    def draw_mask(self, mask: np.ndarray) -> None:
        """Visualize 3D points projected onto mask."""
        if "mask" not in self.stages:
            return
        assert mask.dtype == np.uint8, "Mask must be uint8"

        vis = self.frame_curr
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.5, mask, 0.5, 0)
        self.frame_curr = vis

    def draw_projection(
        self,
        pts_3d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """Visualize 3D points projected onto current frame."""
        if "projection" not in self.stages:
            return
        vis = self.frame_curr
        pts_2d, _ = cv2.projectPoints(pts_3d, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
        pts_2d = pts_2d.reshape(-1, 2)
        for pt in pts_2d:
            max_size = vis.shape[1::-1]
            valid = (pt >= 0).all() & (pt < max_size).all()
            if not valid:
                continue
            center = pt.astype(int)
            bl = (center - np.array([2, 2])).clip(min=0, max=max_size)
            tr = (center + np.array([2, 2])).clip(min=0, max=max_size)
            cv2.rectangle(vis, tuple(bl), tuple(tr), (0, 255, 255), -1)


def optical_flow_pyrlk(prev_frame, frame, pts_old):
    """
    Calculate the optical flow using the PyRLK algorithm.
    Args:
        prev_frame: The previous frame.
        frame: The current frame.
        pts_old: The previous points (N, 2).
    Returns:
        pts_next: The next points.
        status: The status of the points.
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        minEigThreshold=1e-3,
    )

    pts_next, status, errs = cv2.calcOpticalFlowPyrLK(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        pts_old.reshape(-1, 1, 2).astype(np.float32),
        None,
        **lk_params
    )
    pts_next = pts_next.reshape(-1, 2)
    status = status.ravel().astype(bool)

    # filter out the points with large errors with modified z-score
    errs = np.linalg.norm(pts_next - pts_old, axis=-1)
    median = np.median(errs[status])
    d = np.abs(errs[status] - median)
    mad = np.median(d).clip(min=1e-6)
    modified_z_scores = 0.6745 * d / mad
    status[status] = modified_z_scores <= 3.5
    return pts_next, status


@dataclass
class CameraState:
    frame_idx: int
    K: np.ndarray = field(default_factory=lambda: np.eye(3))
    k: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    C: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "CameraState":
        return CameraState(
            frame_idx=self.frame_idx,
            K=self.K.copy(),
            k=self.k.copy(),
            R=self.R.copy(),
            C=self.C.copy(),
        )

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.C

    def get_ypr(self, deg: bool = True) -> Tuple[float, float, float]:
        yaw, pitch, roll = CameraTracker.rotation_matrix_to_euler(self.R)
        if deg:
            return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
        else:
            return yaw, pitch, roll


def extract_lane_lines_mask(image):
    """extract lane lines from the image using adaptive thresholding and masking
    args:
        image: (H, W, 3) - BGR image
    returns:
        mask: (H, W) - mask of the lane lines (np.uint8)
    """
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lightness = image_hls[:, :, 1]

    mask_thin = cv2.adaptiveThreshold(lightness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -10)
    mask_thick = cv2.adaptiveThreshold(lightness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -10)
    mask = cv2.bitwise_or(mask_thin, mask_thick)

    # suppress very dark pixels using grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask[gray < 30] = 0
    return mask


@dataclass
class CameraTrackerOptions:
    refine_interval: int = 10
    debug_stages: Tuple[str, ...] = ("projection",)


class CameraTracker:
    """
    Tracks camera extrinsics using Euler angles (yaw, pitch, roll) and position.

    Designed for broadcast football footage where:
    - Camera position is typically fixed or slowly moving
    - Smooth rotation (pan/tilt) is the primary motion
    - Roll is usually close to zero

    State:
        yaw: Rotation around vertical axis (pan left/right), in radians
        pitch: Rotation around horizontal axis (tilt up/down), in radians
        roll: Rotation around viewing axis (typically ~0), in radians
        x, y, z: Camera position in world coordinates
    """

    def __init__(
        self,
        pitch_points: np.ndarray,
        fps: float = 30.0,
        options: CameraTrackerOptions = CameraTrackerOptions(),
    ):
        """
        Initialize the camera tracker.
        """
        # State: [yaw, pitch, roll, x, y, z]
        self.state = None
        self.velocity = None  # [d_yaw, d_pitch, d_roll, d_x, d_y, d_z]
        self.covariance = None
        self.frame_buffer = deque(maxlen=3)
        self.camera_states = []
        self.pitch_points = pitch_points
        self.refine_interval = options.refine_interval
        self.debug_vis = Debugger(debug_stages=options.debug_stages)

    def initialize(self, frame_idx: int, K: np.ndarray, k: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        """
        Initialize tracker from rotation matrix and translation vector.

        Args:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        C = -R.T @ t
        self.state = CameraState(frame_idx=frame_idx, K=K, k=k, R=R, C=C)

    def track(self, frame_idx: int, frame: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> None:
        # update the state
        self.state.frame_idx = frame_idx
        self.state.K = K
        self.state.k = dist_coeffs
        self.debug_vis.update(frame.copy())

        if frame_idx == 0:
            self._prepare_field_mask(frame)

        refine_mask = frame_idx > 0 and frame_idx % self.refine_interval == 0
        if refine_mask:
            mask = extract_lane_lines_mask(frame)
            field_mask = self._create_field_mask(frame)
            mask = cv2.bitwise_and(mask, field_mask)
            dist_map, labels, label2yx = self._make_dist_map(mask)
        else:
            labels = None
            label2yx = None

        if frame_idx > 0:
            self._update_flow(
                frame=frame,
                prev_frame=self.frame_buffer[-1],
                state_prev=self.camera_states[-1],
                state_curr=self.state,
                dist_labels=labels,
                label2yx=label2yx,
                dist_map=dist_map,
            )

        if refine_mask:
            self.debug_vis.draw_mask(mask)
            self._update_mask_refine(
                dist_map=dist_map,
                state_curr=self.state,
            )

        if self.debug_vis.visualize:
            self.debug_vis.draw_projection(self.pitch_points, self.state.R, self.state.t, self.state.K, self.state.k)
            cv2.imshow("Visualization", self.debug_vis.frame_curr)
            key = cv2.waitKey(1)
            if key == ord("q"):
                exit()

        self.camera_states.append(self.state.copy())
        self.frame_buffer.append(frame)
        return self.state

    def _project_pitch_points(self, K, k, R, t, img_size):
        pts_2d, _ = cv2.projectPoints(self.pitch_points, cv2.Rodrigues(R)[0], t, K, k)
        pts_2d = pts_2d.reshape(-1, 2)
        H, W = img_size[:2]
        valid = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
        return pts_2d, valid
    
    def _prepare_field_mask(self, frame: np.ndarray, dilation_size: int = 20):
        pts_2d_prev, valid = self._project_pitch_points(
            self.state.K, self.state.k, self.state.R, self.state.t, frame.shape[:2]
        )
        hull = cv2.convexHull(pts_2d_prev[valid].astype(np.int32))
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask = cv2.dilate(mask, kernel)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean, std = cv2.meanStdDev(hsv_frame, mask=mask)
        m = mean.flatten()
        s = std.flatten()

        # assume gaussian distribution
        min_h = m[0] - 2.0 * s[0]
        max_h = m[0] + 2.0 * s[0]
        min_s = m[1] - 2.5 * s[1]
        max_s = m[1] + 2.5 * s[1]
        min_v = m[2] - 3.0 * s[2] 
        max_v = m[2] + 3.0 * s[2]
        self.lower_bound = np.array([min_h, min_s, min_v])
        self.upper_bound = np.array([max_h, max_s, max_v])

    def _create_field_mask(self, frame: np.ndarray):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)
        # we want to fill the holes in the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        return mask

    def _update_flow(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        state_prev: CameraState,
        state_curr: CameraState,
        dist_map: np.ndarray | None = None,
        dist_labels: np.ndarray | None = None,
        label2yx: np.ndarray | None = None,
    ) -> CameraState:
        """
        Update camera state from optical flow.
        """
        pts_2d_prev, valid = self._project_pitch_points(
            state_prev.K, state_prev.k, state_prev.R, state_prev.t, frame.shape[:2]
        )

        # snap the points to the nearest mask point
        pts_2d_prev = pts_2d_prev[valid]
        pts_2d_prev_int = pts_2d_prev.astype(np.int32)
        if dist_labels is not None and label2yx is not None or dist_map is not None:
            # ensure the dist is not too far away, too
            labels = dist_labels[pts_2d_prev_int[:, 1], pts_2d_prev_int[:, 0]]
            dist = dist_map[pts_2d_prev_int[:, 1], pts_2d_prev_int[:, 0]]
            pts_2d_prev = label2yx[labels[dist < 20]]
        pts_2d_prev = pts_2d_prev.astype(np.float32)

        pts_2d_next, status = optical_flow_pyrlk(prev_frame, frame, pts_2d_prev)

        self.debug_vis.draw_optical_flow(pts_2d_prev, pts_2d_next, status)

        # estimate the rotation from the optical flow
        pts_2d_prev_normalized = self._prep_points(pts_2d_prev[status], state_prev.K, state_prev.k)
        pts_2d_next_normalized = self._prep_points(pts_2d_next[status], state_curr.K, state_curr.k)

        M = pts_2d_next_normalized.T @ pts_2d_prev_normalized
        U, S, Vt = np.linalg.svd(M)
        R_rel = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        R = R_rel @ state_prev.R

        # update the state
        self.state.R = R

    def _update_mask_refine(
        self,
        dist_map: np.ndarray,
        state_curr: CameraState,
    ) -> CameraState:
        """
        Refine camera rotation by aligning projected 3D points with visual features.

        Args:
            timestamp: float, timestamp of the frame
            dist_map: (H, W) distance transform of visual features (e.g., lane lines)
            state_curr: Current camera state
        """
        R = self._refine_rotation_with_mask(
            dist_map=dist_map,
            pts_3d=self.pitch_points,
            K=state_curr.K,
            R_init=state_curr.R,
            C=state_curr.C,
            dist_coeffs=state_curr.k,
        )

        # Update state with refined rotation
        self.state.R = R

    # ========================================================================
    # Static utility methods for coordinate transformations
    # ========================================================================
    @staticmethod
    def _prep_points(pts, K, dist):
        if dist is not None:
            dist = np.asarray(dist, dtype=np.float32).ravel()
            if dist.size == 2:  # k1,k2 only -> expand
                dist = np.array([dist[0], dist[1], 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            dist = None
        pts_ud = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist, P=None).reshape(-1, 2)
        pts_n = np.c_[pts_ud, np.ones(pts_ud.shape[0])]
        pts_n = pts_n / np.linalg.norm(pts_n, axis=1, keepdims=True)
        return pts_n

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to yaw/pitch/roll using custom camera convention.

        This follows OpenCV camera convention where:
        - R[2] is the forward direction (camera's viewing direction) in world coords
        - R[1] is the up direction in world coords
        - R[0] is the right direction in world coords

        Args:
            R: (3, 3) rotation matrix

        Returns:
            yaw: Rotation in radians (derived from forward direction projection)
            pitch: Rotation in radians (upward angle of camera)
            roll: Rotation in radians (camera tilt/roll)
        """
        assert R.shape == (3, 3)

        f = R[2]
        pitch = np.arcsin(np.clip(f[2], -1.0, 1.0))
        yaw = np.arctan2(f[0], f[1])

        sy, cy = np.sin(yaw), np.cos(yaw)
        r0 = np.array([cy, -sy, 0.0], dtype=np.float64)

        cr = np.dot(R[0], r0)
        sr = np.dot(R[1], r0)
        roll = np.arctan2(sr, cr)
        return yaw, pitch, roll

    @staticmethod
    def find_closest_orthogonal_matrix(A: np.ndarray) -> np.ndarray:
        """
        Find closest orthogonal matrix to A in terms of Frobenius norm.

        Args:
            A: (3, 3) matrix

        Returns:
            Q: (3, 3) orthogonal matrix closest to A
        """
        U, _, Vt = np.linalg.svd(A)
        return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    # ========================================================================
    # Private methods - integration with existing algorithms
    # ========================================================================
    def _snap_points_to_mask(self, pts_2d: np.ndarray, dist_map: np.ndarray) -> np.ndarray:
        """snap the points to the nearest mask point"""
        H, W = dist_map.shape[:2]
        xs = np.round(pts_2d[:, 0]).astype(np.int32)
        ys = np.round(pts_2d[:, 1]).astype(np.int32)
        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        return pts_2d[valid]

    def _refine_rotation_with_mask(
        self,
        dist_map: np.ndarray,
        pts_3d: np.ndarray,
        K: np.ndarray,
        R_init: np.ndarray,
        C: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Refine rotation matrix by minimizing distance to visual features.
        This is a refactored version of your original refine_camera_rotation() function.
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5, dtype=np.float32)
        H, W = dist_map.shape[:2]

        # Define objective function
        def objective_function(delta):
            R = (delta + r_init).reshape(3, 3)
            R = self.find_closest_orthogonal_matrix(R)
            t = -R @ C

            pts_2d, _ = cv2.projectPoints(pts_3d, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
            pts_2d = pts_2d.squeeze(axis=1)
            pts_2d = pts_2d.clip([-500, -500], [W + 500, H + 500])

            xs = np.round(pts_2d[:, 0]).astype(np.int32)
            ys = np.round(pts_2d[:, 1]).astype(np.int32)

            valid_mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            xs_valid = xs[valid_mask]
            ys_valid = ys[valid_mask]

            if len(xs_valid) == 0:
                distances = np.sqrt(H**2 + W**2)
            else:
                distances = dist_map[ys_valid, xs_valid]
                distances = distances.mean()
            return distances

        # Optimize
        r_init = R_init.flatten()
        r_delta = np.zeros_like(r_init)

        epsilon = 0.2
        lower_bounds = r_delta - epsilon
        upper_bounds = r_delta + epsilon

        result = scipy.optimize.least_squares(
            objective_function,
            r_delta,
            method="trf",
            bounds=(lower_bounds, upper_bounds),
        )

        # Enforce orthogonality
        R_refined = (result.x + r_init).reshape(3, 3)
        R_refined = self.find_closest_orthogonal_matrix(R_refined)
        return R_refined

    @staticmethod
    def _make_dist_map(mask: np.ndarray) -> np.ndarray:
        """Create distance transform from binary mask."""
        mask_inv = (1 - (mask > 0)).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            mask_inv, cv2.DIST_L2, maskSize=11, labelType=cv2.DIST_LABEL_PIXEL
        )
        ys, xs = np.where(mask_inv == 0)
        seed_labels = labels[ys, xs].astype(np.int32)
        L = int(seed_labels.max()) + 1
        label2yx = np.zeros((L, 2), dtype=np.int32)
        label2yx[seed_labels, 0] = xs
        label2yx[seed_labels, 1] = ys
        return dist, labels, label2yx