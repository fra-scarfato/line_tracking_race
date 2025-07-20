import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from ament_index_python import get_package_share_directory

DEBUG = False  # Toggle verbose visual debugging

class TrackUtility:
    """
    Singleton class to process a racetrack image and extract its centerline,
    transform it into world coordinates, and compute geometric properties like
    curvature and tangents along the track.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Ensure only one instance exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        """Run initialization only once, even across multiple instantiations."""
        # Step 1: Extract centerline pixels from image
        pkg_path  = get_package_share_directory("line_tracking_race_description")
        img_path  = os.path.join(pkg_path, "models/line_track/materials/textures/track_loop.png")
        pixel_pts = self._extract_centerline_pixels(img_path)

        # Step 2: Convert pixel coordinates to local track coordinates (meters)
        track_pts = self._pixels_to_local_frame(pixel_pts)

        # Step 3: Apply static rotation and translation to convert to world frame
        x0, y0, yaw = 1.0, 2.3, 0.0
        self.world_points = self._apply_static_transform(track_pts, x0, y0, yaw)

        # Step 4: Compute geometric properties along the centerline
        self._compute_curvilinear_properties()

    # ----------- Step 1: Image → Skeleton Pixels -----------
    def _extract_centerline_pixels(self, image_path):
        """
        Skeletonizes the track area to find a 1-pixel wide centerline.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")

        # Convert to grayscale and threshold to create binary image
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Extract contours to define outer and inner track boundaries
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            raise RuntimeError("Expected at least 2 contours for track boundaries")

        # Sort contours by area to identify outer and inner boundaries
        areas_and_contours = sorted(
            ((cv2.contourArea(c), c) for c in contours),
            key=lambda x: x[0]
        )
        inner_contour, outer_contour = areas_and_contours[0][1], areas_and_contours[-1][1]

        # Create masks for inner and outer contours and XOR them to get track ribbon
        h, w = gray.shape
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_outer, [outer_contour], 255)
        cv2.fillPoly(mask_inner, [inner_contour], 255)
        track_mask = cv2.bitwise_xor(mask_outer, mask_inner)

        # Skeletonize the ribbon mask → 1-pixel wide path
        skeleton_bool = track_mask.astype(bool)
        skeleton      = skeletonize(skeleton_bool)

        # Convert skeleton pixels to (x, y) and sort into a connected path
        pixels = np.column_stack(np.where(skeleton))
        pixels = pixels[:, [1, 0]]  # from (row, col) → (x, y)
        sorted_pts = self._sort_skeleton(pixels, skeleton)

        # Visual debug if enabled
        if DEBUG:
            self._debug_visualize(img, outer_contour, inner_contour, skeleton, sorted_pts)

        return sorted_pts

    def _sort_skeleton(self, pts, skeleton):
        """
        Sort unordered skeleton pixels into a continuous path based on 8-connected neighbors.
        """
        if len(pts) < 2:
            return pts.copy()

        # Map from pixel to neighbors using 8-connectivity
        skeleton_coords = set(map(tuple, pts))
        neighbors = {}
        for x, y in pts:
            nbrs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in skeleton_coords:
                        nbrs.append((nx, ny))
            neighbors[(x, y)] = nbrs

        # Start from endpoint (pixel with fewest neighbors)
        start = min(neighbors, key=lambda p: len(neighbors[p]))
        path, visited = [start], {start}

        # Follow neighbors until all points visited
        while len(path) < len(pts):
            curr = path[-1]
            unvisited = [n for n in neighbors[curr] if n not in visited]
            if unvisited:
                # Directional continuity (keep same direction if multiple options)
                next_pt = unvisited[0]
                if len(unvisited) > 1 and len(path) > 1:
                    prev_dir = np.array(curr) - np.array(path[-2])
                    prev_dir = prev_dir / (np.linalg.norm(prev_dir) + 1e-8)
                    scores = [
                        np.dot(prev_dir, (np.array(cand) - np.array(curr)) /
                               (np.linalg.norm(np.array(cand) - np.array(curr)) + 1e-8))
                        for cand in unvisited
                    ]
                    next_pt = unvisited[int(np.argmax(scores))]
                path.append(next_pt)
                visited.add(next_pt)
            else:
                # Jump to nearest unvisited pixel if stuck
                remaining = [tuple(p) for p in pts if tuple(p) not in visited]
                dists = [np.hypot(r[0]-curr[0], r[1]-curr[1]) for r in remaining]
                nearest = remaining[int(np.argmin(dists))]
                path.append(nearest)
                visited.add(nearest)

        return np.array(path, dtype=int)

    def _debug_visualize(self, img, outer, inner, skeleton, centerline_pts):
        """Debug visualization: original image, contours, skeleton, and centerline."""
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.ravel()

        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title("Original")

        cont_img = img.copy()
        cv2.drawContours(cont_img, [outer], -1, (0,255,0), 2)
        cv2.drawContours(cont_img, [inner], -1, (255,0,0), 2)
        axs[1].imshow(cv2.cvtColor(cont_img, cv2.COLOR_BGR2RGB)); axs[1].set_title("Contours")

        axs[2].imshow(skeleton, cmap="gray"); axs[2].set_title("Skeleton")

        cl_img = img.copy()
        for p in centerline_pts:
            cv2.circle(cl_img, tuple(p), 1, (0,0,255), -1)
        axs[3].imshow(cv2.cvtColor(cl_img, cv2.COLOR_BGR2RGB)); axs[3].set_title("Centerline")

        for ax in axs: ax.axis("off")
        plt.tight_layout(); plt.show()

    # ----------- Step 2: Pixel → Local Frame -----------
    def _pixels_to_local_frame(self, pixel_pts, image_size=1024, track_diameter=10.0):
        """
        Convert pixel positions to metric space.
        The origin is set at the center of the image.
        """
        scale = track_diameter / image_size  # meters per pixel
        return [
            ((x - image_size/2)*scale, (image_size/2 - y)*scale)
            for x, y in pixel_pts
        ]

    # ----------- Step 3: Local → World Transform -----------
    def _apply_static_transform(self, pts, x0, y0, yaw):
        """
        Apply rotation (yaw) and then translation (x0, y0) to local frame points.
        Standard rigid body transform in 2D.
        """
        c, s = np.cos(yaw), np.sin(yaw)
        return [
            (c*x - s*y + x0, s*x + c*y + y0)
            for x, y in pts
        ]

    # ----------- Step 4: Arc-Length, Tangents, Curvature -----------
    def _compute_curvilinear_properties(self, smooth_win=5):
        """
        Compute:
        - s: Arc-length from start along path
        - Tangents: unit direction vectors at each point
        - Curvature: rate of change of direction along the curve
        """
        pts = np.array(self.world_points)
        N = len(pts)
        if N < 3:
            raise RuntimeError("Need ≥3 points for curvilinear properties")

        # Arc-length calculation (cumulative distances between points)
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self.curvilinear_abscissa = np.concatenate([[0.0], np.cumsum(dists)])

        # Initialize tangent and curvature arrays
        self.tangent_vectors = np.zeros((N, 2), float)
        self.curvature = np.zeros(N, float)

        # Compute tangents and curvature using local fits
        self._fit_tangents(pts, smooth_win)
        self._fit_curvature(pts, smooth_win)

        if DEBUG:
            self._visualize_curvilinear()

    def _fit_tangents(self, pts, w):
        """Fit tangent vectors by linear polynomial regression in a local window."""
        N = len(pts)
        half = w // 2
        padded = np.vstack([pts[-half:], pts, pts[:half]])
        t = np.arange(-half, half+1)

        for i in range(N):
            window = padded[i:i + w]
            dx = -np.polyfit(t, window[:, 0], 1)[0]
            dy = -np.polyfit(t, window[:, 1], 1)[0]
            vec = np.array([dx, dy])
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                vec = (pts[(i+1)%N] - pts[i])
                norm = np.linalg.norm(vec) or 1.0
            self.tangent_vectors[i] = vec / norm

    def _fit_curvature(self, pts, w):
        """Estimate curvature as rate of change of tangent vector with arc-length."""
        N = len(pts)
        half = w // 2
        tang = self.tangent_vectors
        pad = np.vstack([tang[-half:], tang, tang[:half]])

        for i in range(N):
            ds1 = np.linalg.norm(pts[i] - pts[i-1]) if i > 0 else np.linalg.norm(pts[0] - pts[-1])
            ds2 = np.linalg.norm(pts[(i+1)%N] - pts[i])
            ds = (ds1 + ds2) / 2.0 or 1.0

            # Central difference on tangent vectors to get dT/ds
            dT = (pad[i+half+1] - pad[i+half-1]) / (2 * ds)
            self.curvature[i] = np.linalg.norm(dT)

    def _visualize_curvilinear(self):
        """Visualize centerline with tangent vectors and curvature coloring."""
        pts = np.array(self.world_points)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Centerline and tangents
        ax1.plot(pts[:, 0], pts[:, 1], '-b')
        for i in range(0, len(pts), max(1, len(pts)//30)):
            p = pts[i]
            d = self.tangent_vectors[i] * 0.5
            ax1.arrow(p[0], p[1], d[0], d[1], head_width=0.1, head_length=0.1)
        ax1.set_title("Centerline & Tangents"); ax1.axis("equal")

        # Color-coded curvature
        sc = ax2.scatter(pts[:, 0], pts[:, 1], c=self.curvature, cmap="plasma", s=10)
        fig.colorbar(sc, ax=ax2, label="Curvature (1/m)")
        ax2.set_title("Centerline Colored by Curvature"); ax2.axis("equal")

        plt.tight_layout(); plt.show()

    # ----------- Public Accessors -----------
    def get_world_points(self): return self.world_points
    def get_curvilinear_abscissa(self): return self.curvilinear_abscissa
    def get_tangent_vectors(self): return self.tangent_vectors
    def get_curvature(self): return self.curvature

    def get_normal_vectors(self):
        """Normal vectors are perpendicular to tangent vectors (90° CCW)."""
        return np.column_stack([-self.tangent_vectors[:, 1],
                                self.tangent_vectors[:, 0]])

    def interpolate_at_arc_length(self, s_target):
        """
        Interpolate centerline properties at arbitrary arc-length `s_target`.
        Wraps around on closed track.
        """
        raise NotImplementedError("Use your existing implementation")

# ----------- Run Standalone Debug Test -----------
if __name__ == "__main__":
    DEBUG = True
    tu = TrackUtility()
    pts = tu.get_world_points()
    s = tu.get_curvilinear_abscissa()
    print(f"Track length = {s[-1]:.2f} m, points = {len(pts)}")
    start_point = pts[0]
    print(f"Starting point (world coords): x = {start_point[0]:.3f}, y = {start_point[1]:.3f}")