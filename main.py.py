import math
import heapq
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# AYARLAR
# =========================================================
ROWS, COLS = 120, 120
STEP_LENGTH = 1.0
TURN_ANGLE = 30.0
NUM_HEADINGS = 12
TERRAIN_DENSITY_SCALE = 1.5

SENSOR_RADIUS = 9
UNKNOWN_TERRAIN_COST = 2.8
UNKNOWN_SLOPE_ESTIMATE = 0.08

START = None
GOAL = None

# Terrain türleri
FLAT = 0
ROCK = 1
CRATER = 2
STEEP = 3
ICE = 4
SOFT = 5
UNKNOWN = -1

TERRAIN_NAMES = {
    FLAT: "FLAT",
    ROCK: "ROCK",
    CRATER: "CRATER",
    STEEP: "STEEP",
    ICE: "ICE",
    SOFT: "SOFT",
    UNKNOWN: "UNKNOWN"
}

TERRAIN_COLORS = {
    FLAT: "#46d36b",
    ROCK: "#f2f2f2",
    CRATER: "#4a4a4a",
    STEEP: "#ffd84d",
    ICE: "#6ec5ff",
    SOFT: "#b7845a",
    UNKNOWN: "#d9dde6"
}

legend_desc = {
    FLAT: "Normal / Safe",
    ICE: "Slip Risk",
    SOFT: "Sink Risk",
    STEEP: "Tilt Risk",
    ROCK: "Blocked",
    CRATER: "Danger",
    UNKNOWN: "Unexplored"
}

DECISION_INTERVAL = 40
DECISION_LOOKAHEAD = 10

# =========================================================
# VALIDATION + ENERGY + MOTION PARAMETRELERİ
# =========================================================
VALIDATION_RADIUS = 5
CLEARANCE_SEARCH_RADIUS = 8
OBSTACLE_INFLATION_RADIUS = 1

POINT_RULES = {
    "slope_warn": 0.15,
    "slope_reject": 0.24,
    "unsafe_ratio_warn": 0.22,
    "unsafe_ratio_reject": 0.34,
    "clearance_warn": 2.0,
    "clearance_reject": 1.2,
}

ENERGY_MODEL = {
    "base_step": 0.18,
    "terrain_bonus": {
        FLAT: 0.00,
        ICE: 0.10,
        SOFT: 0.16,
        STEEP: 0.20,
    },
    "slope_coeff": 0.95,
    "turn_coeff": 0.06,
    "reserve_margin": 0.15,
}

MOTION_SUBSTEPS = 5
FRAME_INTERVAL_MS = 120

REPLAN_LOOKAHEAD = 18
REPLAN_COOLDOWN_SEC = 0.60

# donma fix parametreleri
REPLAN_MIN_PROGRESS_STEPS = 2
REPLAN_FAIL_IGNORE_SEC = 1.20

# retreat / backtrack
safe_history = []
RETREAT_MIN_ANCHOR_GAP = 6
MAX_RETREAT_POINTS = 80

mission_report = None
motion_substep = 0

active_sensor_snapshot = []
view_mode = "planning"
planning_overlay_img = None
orbital_ax = None
orbital_rover_marker = None
orbital_start_marker = None
orbital_goal_marker = None
orbital_trail_line = None
lidar_ax = None
lidar_status_text = None
lidar_points_scatter = None
lidar_last_points = []
lidar_last_pose = None
lidar_last_scan_time = -1e9
last_replan_time = -1e9
last_replan_fail_time = -1e9
last_replan_fail_cell = None

# planner için gizli engel haritası
blocked_map = None

# rover world-model (partial observability)
known_terrain = None
known_slope = None
known_depth = None
known_blocked_map = None
explored_mask = None
current_visibility_mask = None
fog_overlay_img = None
sensor_ring = None

# yeni yoğunluk / tehlike haritaları
ice_strength_map = None
soft_sink_map = None
hazard_blocked_map = None
depth_map = None
crater_bowl_map = None
crater_rim_map = None

# clearance cache
clearance_cache = {}

# =========================================================
# YARDIMCI
# =========================================================
def wrap_heading(h):
    return h % NUM_HEADINGS

def heading_to_deg(h):
    return h * TURN_ANGLE

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def heuristic(x, y, gx, gy):
    return math.hypot(gx - x, gy - y)

def nearest_cell(x, y):
    return int(round(y)), int(round(x))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def scaled_int_count(value):
    return max(1, int(round(value * TERRAIN_DENSITY_SCALE)))

def scaled_int_range(lo, hi):
    a = max(1, int(round(lo * TERRAIN_DENSITY_SCALE)))
    b = max(a, int(round(hi * TERRAIN_DENSITY_SCALE)))
    return a, b

def block_average_rgb(img, factor=6):
    rows, cols, ch = img.shape
    out = img.copy()
    for r in range(0, rows, factor):
        for c in range(0, cols, factor):
            block = img[r:min(rows, r + factor), c:min(cols, c + factor)]
            avg = block.mean(axis=(0, 1))
            out[r:min(rows, r + factor), c:min(cols, c + factor)] = avg
    return out

def make_orbital_rgb(base_rgb, terrain_map, depth_local):
    base_rgb = base_rgb[..., :3]
    orbital = block_average_rgb(base_rgb, factor=6)
    orbital = 0.78 * orbital + 0.22

    major_overlay = np.zeros((ROWS, COLS, 4), dtype=float)
    major_overlay[terrain_map == ROCK] = [0.92, 0.92, 0.92, 0.28]
    major_overlay[terrain_map == CRATER] = [0.12, 0.12, 0.12, 0.30]
    major_overlay[terrain_map == STEEP] = [0.92, 0.82, 0.32, 0.15]

    rgb = orbital.copy()
    alpha = major_overlay[..., 3:4]
    rgb = rgb * (1.0 - alpha) + major_overlay[..., :3] * alpha

    depth_norm = np.clip(depth_local / 100.0, 0.0, 1.0)
    rgb *= (1.0 - 0.10 * depth_norm[..., None])
    return np.clip(rgb, 0, 1)

def initialize_known_maps():
    global known_terrain, known_slope, known_depth, known_blocked_map, explored_mask, current_visibility_mask
    known_terrain = np.full((ROWS, COLS), UNKNOWN, dtype=int)
    known_slope = np.full((ROWS, COLS), np.nan, dtype=float)
    known_depth = np.full((ROWS, COLS), np.nan, dtype=float)
    known_blocked_map = np.zeros((ROWS, COLS), dtype=bool)
    explored_mask = np.zeros((ROWS, COLS), dtype=bool)
    current_visibility_mask = np.zeros((ROWS, COLS), dtype=bool)

def build_known_blocked_map(known_terrain_map):
    base = (known_terrain_map == ROCK) | (known_terrain_map == CRATER)
    return inflate_obstacles(base, radius=OBSTACLE_INFLATION_RADIUS)

def reveal_local_area(x, y, radius=SENSOR_RADIUS):
    global known_terrain, known_slope, known_depth, known_blocked_map, explored_mask, current_visibility_mask
    row, col = nearest_cell(x, y)
    current_visibility_mask[:] = False

    prev_known_blocked = known_blocked_map.copy()
    newly_revealed = 0

    for rr, cc in circular_cells(row, col, radius):
        if known_terrain[rr, cc] == UNKNOWN:
            newly_revealed += 1
        known_terrain[rr, cc] = terrain[rr, cc]
        known_slope[rr, cc] = slope_map[rr, cc]
        known_depth[rr, cc] = depth_map[rr, cc]
        explored_mask[rr, cc] = True
        current_visibility_mask[rr, cc] = True

    known_blocked_map = build_known_blocked_map(known_terrain)
    new_blocked = bool(np.any(known_blocked_map & (~prev_known_blocked)))
    return {
        "new_cells": newly_revealed,
        "new_blocked": new_blocked,
    }

def planner_terrain_map():
    if known_terrain is None:
        return terrain
    planner_map = known_terrain.copy()
    planner_map[planner_map == UNKNOWN] = FLAT
    return planner_map

def planner_slope_map():
    if known_slope is None:
        return slope_map
    return np.where(np.isnan(known_slope), UNKNOWN_SLOPE_ESTIMATE, known_slope)

def refresh_visibility_overlay():
    if fog_overlay_img is None:
        return
    fog = np.ones((ROWS, COLS, 4), dtype=float)
    fog[..., 0] = 0.96
    fog[..., 1] = 0.97
    fog[..., 2] = 0.99
    fog[..., 3] = 0.88
    if explored_mask is not None:
        fog[explored_mask, 3] = 0.35
    if current_visibility_mask is not None:
        fog[current_visibility_mask, 3] = 0.02
    fog_overlay_img.set_data(fog)
    fog_overlay_img.set_visible(view_mode == "rover")

    if sensor_ring is not None and START is not None and selection_stage > 0 and view_mode == "rover":
        if sim_state == "moving" and trail_x and trail_y:
            cx, cy = trail_x[-1], trail_y[-1]
        elif path_states:
            idx = min(max(current_index, 0), len(path_states) - 1)
            cx, cy = path_states[idx][0], path_states[idx][1]
        else:
            cx, cy = START
        sensor_ring.center = (cx, cy)
        sensor_ring.set_radius(SENSOR_RADIUS)
        sensor_ring.set_visible(True)
    elif sensor_ring is not None:
        sensor_ring.set_visible(False)

def set_main_view(mode):
    global view_mode
    view_mode = mode
    if planning_overlay_img is not None:
        planning_overlay_img.set_visible(mode == "planning")
    refresh_visibility_overlay()
    update_orbital_inset()
    update_orbital_inset()

def update_orbital_inset():
    if orbital_start_marker is not None:
        if START is not None:
            orbital_start_marker.set_data([START[0]], [START[1]])
        else:
            orbital_start_marker.set_data([], [])

    if orbital_goal_marker is not None:
        if GOAL is not None:
            orbital_goal_marker.set_data([GOAL[0]], [GOAL[1]])
        else:
            orbital_goal_marker.set_data([], [])

    if orbital_rover_marker is not None:
        if START is None:
            orbital_rover_marker.set_data([], [])
        elif sim_state == "moving" and trail_x and trail_y:
            orbital_rover_marker.set_data([trail_x[-1]], [trail_y[-1]])
        elif path_states:
            idx = min(max(current_index, 0), len(path_states) - 1)
            orbital_rover_marker.set_data([path_states[idx][0]], [path_states[idx][1]])
        else:
            orbital_rover_marker.set_data([START[0]], [START[1]])

    if orbital_trail_line is not None:
        orbital_trail_line.set_data(trail_x, trail_y)

    if orbital_ax is not None:
        title = "MISSION PLANNING MAP" if view_mode == "planning" else "ORBITAL OVERVIEW"
        orbital_ax.set_title(title, fontsize=7, pad=2, color="white")

def terrain_label_for_display(row, col):
    if known_terrain is not None and not explored_mask[row, col]:
        return UNKNOWN
    return terrain[row, col]


def terrain_rgb_for_lidar(terrain_type):
    hex_color = TERRAIN_COLORS.get(int(terrain_type), "#d9dde6")
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0.85, 0.87, 0.90)
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def perform_lidar_scan(x, y, heading_idx, max_range=None, num_rays=31, step=0.55):
    if max_range is None:
        max_range = SENSOR_RADIUS + 5.0

    heading_rad = deg_to_rad(heading_to_deg(heading_idx))
    half_fov = math.radians(85.0)

    points = []
    hit_count = 0

    for rel_angle in np.linspace(-half_fov, half_fov, num_rays):
        ray_angle = heading_rad + rel_angle
        distance = step

        while distance <= max_range:
            wx = x + math.cos(ray_angle) * distance
            wy = y + math.sin(ray_angle) * distance

            if wx < 0 or wx >= COLS or wy < 0 or wy >= ROWS:
                break

            rr, cc = nearest_cell(wx, wy)
            rr = max(0, min(ROWS - 1, rr))
            cc = max(0, min(COLS - 1, cc))

            terrain_type = terrain[rr, cc]
            terrain_height = float(heightmap[rr, cc]) * 12.0
            base_height = float(heightmap[nearest_cell(x, y)[0], nearest_cell(x, y)[1]]) * 12.0
            rel_z = terrain_height - base_height

            is_obstacle = terrain_type in (ROCK, CRATER) or slope_map[rr, cc] > POINT_RULES["slope_warn"]

            if is_obstacle:
                forward = distance * math.cos(rel_angle)
                lateral = distance * math.sin(rel_angle)
                points.append((lateral, forward, rel_z, terrain_type))
                hit_count += 1
                break

            # seyrek zemin noktaları da göster, kutu boş görünmesin
            if int(distance / step) % 5 == 0:
                forward = distance * math.cos(rel_angle)
                lateral = distance * math.sin(rel_angle)
                points.append((lateral, forward, rel_z * 0.55, terrain_type))

            distance += step

    return points, hit_count

def update_lidar_panel(x=None, y=None, heading_idx=None):
    global lidar_points_scatter, lidar_status_text, lidar_last_points, lidar_last_pose, lidar_last_scan_time

    if lidar_ax is None:
        return

    lidar_ax.cla()
    lidar_ax.set_facecolor("#0b1733")
    lidar_ax.set_title("LIDAR POINT CLOUD", fontsize=7, pad=2, color="white")
    lidar_ax.set_xlim(-10, 10)
    lidar_ax.set_ylim(0, SENSOR_RADIUS + 6)
    lidar_ax.set_zlim(-4, 8)
    lidar_ax.view_init(elev=24, azim=-92)
    lidar_ax.grid(True, alpha=0.18)
    lidar_ax.set_xticks([])
    lidar_ax.set_yticks([])
    lidar_ax.set_zticks([])

    try:
        lidar_ax.xaxis.pane.set_facecolor((0.05, 0.09, 0.18, 1.0))
        lidar_ax.yaxis.pane.set_facecolor((0.05, 0.09, 0.18, 1.0))
        lidar_ax.zaxis.pane.set_facecolor((0.05, 0.09, 0.18, 1.0))
    except Exception:
        pass

    if x is None or y is None or heading_idx is None:
        lidar_ax.text2D(0.08, 0.10, "STANDBY", transform=lidar_ax.transAxes,
                        color="#8ad7ff", fontsize=8, fontweight="bold")
        return

    now = time.perf_counter()
    pose = (round(x, 1), round(y, 1), int(heading_idx))

    should_rescan = False
    if lidar_last_pose is None:
        should_rescan = True
    else:
        moved = math.hypot(x - lidar_last_pose[0], y - lidar_last_pose[1])
        turned = abs(int(heading_idx) - int(lidar_last_pose[2]))
        turned = min(turned, NUM_HEADINGS - turned)
        if moved >= 0.45 or turned >= 1:
            should_rescan = True
        if now - lidar_last_scan_time >= 0.18:
            should_rescan = True

    if should_rescan:
        points, hit_count = perform_lidar_scan(x, y, heading_idx)
        lidar_last_points = points
        lidar_last_pose = pose
        lidar_last_scan_time = now
    else:
        points = lidar_last_points
        hit_count = sum(1 for p in points if p[3] in (ROCK, CRATER, STEEP))

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        cs = [terrain_rgb_for_lidar(p[3]) for p in points]
        ss = [18 if p[3] in (ROCK, CRATER) else 10 for p in points]
        lidar_points_scatter = lidar_ax.scatter(xs, ys, zs, c=cs, s=ss, depthshade=True, alpha=0.92)

    lidar_ax.scatter([0], [0], [0], c=["orange"], s=42, depthshade=False)
    lidar_ax.plot([0, 0], [0, SENSOR_RADIUS * 0.8], [0, 0], color="#8ad7ff", linewidth=1.2, alpha=0.9)

    arc_angles = np.linspace(-math.pi/2.8, math.pi/2.8, 80)
    arc_x = np.sin(arc_angles) * SENSOR_RADIUS
    arc_y = np.cos(arc_angles) * SENSOR_RADIUS
    arc_z = np.zeros_like(arc_x)
    lidar_ax.plot(arc_x, arc_y, arc_z, color="white", linewidth=0.9, alpha=0.35)

    lidar_ax.text2D(
        0.05, 0.06,
        f"hits={hit_count}  rays=31  range={SENSOR_RADIUS + 5:.0f}m",
        transform=lidar_ax.transAxes,
        color="#8ad7ff",
        fontsize=7.5,
        fontweight="bold"
    )

# =========================================================
# RISK TABLOSU
# =========================================================
def terrain_risk_breakdown(terrain_type, row=None, col=None):
    slip = 5
    sink = 5
    tilt = 5
    overall = 8
    warning = "Normal navigation"

    if terrain_type == ICE:
        ice_level = 0.0 if row is None else float(ice_strength_map[row, col])
        depth_level = 0.0 if row is None else float(depth_map[row, col] / 100.0)

        slip = int(50 + 45 * ice_level)
        sink = int(8 + 8 * depth_level)
        tilt = int(12 + 18 * depth_level)
        overall = int(min(98, 40 + 35 * ice_level + 20 * depth_level))

        if ice_level > 0.76 and depth_level > 0.54:
            warning = "Severe ice zone - entry not recommended"
        elif ice_level > 0.65:
            warning = "High ice severity detected"
        else:
            warning = "Slippery surface detected"

    elif terrain_type == SOFT:
        sink_level = 0.0 if row is None else float(soft_sink_map[row, col])
        depth_level = 0.0 if row is None else float(depth_map[row, col] / 100.0)

        slip = int(10 + 10 * sink_level)
        sink = int(40 + 50 * sink_level)
        tilt = int(12 + 16 * depth_level)
        overall = int(min(98, 35 + 38 * sink_level + 18 * depth_level))

        if sink_level > 0.75 and depth_level > 0.49:
            warning = "Severe sink risk - entry not recommended"
        elif sink_level > 0.65:
            warning = "High sink risk detected"
        else:
            warning = "Soft soil detected"

    elif terrain_type == STEEP:
        local_slope = 0.0 if row is None else float(slope_map[row, col])
        slope_norm = local_slope / (float(np.max(slope_map)) + 1e-9)

        slip = int(10 + 10 * slope_norm)
        sink = 10
        tilt = int(55 + 40 * slope_norm)
        overall = int(min(98, 45 + 40 * slope_norm))
        warning = "High slope detected"

    elif terrain_type == ROCK:
        slip, sink, tilt, overall = 10, 10, 15, 95
        warning = "Rock obstacle"

    elif terrain_type == CRATER:
        depth_level = 0.0 if row is None else float(depth_map[row, col] / 100.0)
        slip = int(15 + 10 * depth_level)
        sink = int(20 + 20 * depth_level)
        tilt = int(25 + 25 * depth_level)
        overall = int(min(99, 78 + 20 * depth_level))
        warning = "Crater danger"

    elif terrain_type == UNKNOWN:
        slip, sink, tilt, overall = 20, 20, 20, 35
        warning = "Unexplored terrain - uncertainty penalty"

    else:
        slip, sink, tilt, overall = 5, 5, 8, 12
        warning = "Terrain stable"

    return slip, sink, tilt, overall, warning

# =========================================================
# AY FOTOĞRAFI GİBİ HEIGHTMAP + EK ZEMİNLER
# =========================================================
def generate_lunar_heightmap(rows, cols):
    h = np.zeros((rows, cols), dtype=float)
    yy, xx = np.mgrid[0:rows, 0:cols]

    for _ in range(10):
        cx = random.uniform(0, cols)
        cy = random.uniform(0, rows)
        sigma = random.uniform(18, 40)
        amp = random.uniform(-0.8, 0.8)
        h += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))

    crater_mask = np.zeros((rows, cols), dtype=bool)
    crater_bowl = np.zeros((rows, cols), dtype=float)
    crater_rim = np.zeros((rows, cols), dtype=float)

    crater_lo, crater_hi = scaled_int_range(22, 32)
    for _ in range(random.randint(crater_lo, crater_hi)):
        cx = random.uniform(10, cols - 10)
        cy = random.uniform(10, rows - 10)
        radius = random.uniform(3, 8)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bowl = np.exp(-(dist ** 2) / (2 * (radius * 0.62) ** 2))
        inner_shadow = np.exp(-(dist ** 2) / (2 * (radius * 0.42) ** 2))
        rim = np.exp(-((dist - radius) ** 2) / (2 * (radius * 0.16) ** 2))

        depth = random.uniform(0.8, 1.7)
        rim_height = random.uniform(0.18, 0.52)

        h -= depth * bowl
        h -= 0.18 * depth * inner_shadow
        h += rim_height * rim

        crater_bowl += depth * (0.72 * bowl + 0.28 * inner_shadow)
        crater_rim += rim_height * rim
        crater_mask |= dist <= radius * 0.84

    rock_mask = np.zeros((rows, cols), dtype=bool)
    rock_lo, rock_hi = scaled_int_range(45, 75)
    for _ in range(random.randint(rock_lo, rock_hi)):
        cx = random.uniform(0, cols)
        cy = random.uniform(0, rows)
        sigma = random.uniform(0.8, 1.6)
        amp = random.uniform(0.35, 0.8)

        bump = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        h += bump
        rock_mask |= bump > (amp * 0.68)

    h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    crater_bowl = crater_bowl / (crater_bowl.max() + 1e-9)
    crater_rim = crater_rim / (crater_rim.max() + 1e-9)

    gy, gx = np.gradient(h)
    slope_tmp = np.hypot(gx, gy)

    low_zone = h < np.percentile(h, 32)
    flat_zone = slope_tmp < np.percentile(slope_tmp, 35)

    ice_seed = np.zeros((rows, cols), dtype=float)
    for _ in range(scaled_int_count(6)):
        cx = random.uniform(10, cols - 10)
        cy = random.uniform(10, rows - 10)
        sigma = random.uniform(6, 11)
        ice_seed += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))

    ice_seed = ice_seed / (ice_seed.max() + 1e-9)
    ice_mask = (ice_seed > np.percentile(ice_seed, 83)) & low_zone & flat_zone

    ice_strength = (
        0.60 * ice_seed +
        0.22 * (1.0 - slope_tmp / (slope_tmp.max() + 1e-9)) +
        0.18 * (1.0 - h)
    )
    ice_strength = np.clip(ice_strength, 0.0, 1.0)
    ice_strength *= ice_mask.astype(float)

    soft_seed = np.zeros((rows, cols), dtype=float)
    for _ in range(scaled_int_count(8)):
        cx = random.uniform(0, cols)
        cy = random.uniform(0, rows)
        sigma = random.uniform(6, 14)
        soft_seed += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))

    soft_seed = soft_seed / (soft_seed.max() + 1e-9)
    mid_low = h < np.percentile(h, 55)
    soft_mask = (soft_seed > np.percentile(soft_seed, 81)) & mid_low & (~ice_mask)

    soft_sink = (
        0.56 * soft_seed +
        0.28 * (1.0 - h) +
        0.16 * (1.0 - slope_tmp / (slope_tmp.max() + 1e-9))
    )
    soft_sink = np.clip(soft_sink, 0.0, 1.0)
    soft_sink *= soft_mask.astype(float)

    return h, crater_mask, rock_mask, ice_mask, soft_mask, ice_strength, soft_sink, crater_bowl, crater_rim

# =========================================================
# TERRAIN HARİTASI
# =========================================================
def build_terrain(heightmap, crater_mask, rock_mask, ice_mask, soft_mask):
    gy, gx = np.gradient(heightmap)
    slope = np.hypot(gx, gy)

    terrain_local = np.zeros_like(heightmap, dtype=int)
    terrain_local[:] = FLAT
    terrain_local[crater_mask] = CRATER
    terrain_local[rock_mask] = ROCK

    steep_zone = (slope > np.percentile(slope, 82)) & (terrain_local == FLAT)
    terrain_local[steep_zone] = STEEP

    terrain_local[(terrain_local == FLAT) & ice_mask] = ICE
    terrain_local[(terrain_local == FLAT) & soft_mask] = SOFT

    return terrain_local, slope

# =========================================================
# GİZLİ ENGEL HARİTASI
# =========================================================
def inflate_obstacles(blocked, radius=1):
    if radius <= 0:
        return blocked.copy()

    inflated = blocked.copy()
    rows, cols = blocked.shape

    for r in range(rows):
        for c in range(cols):
            if blocked[r, c]:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            inflated[rr, cc] = True
    return inflated

def build_blocked_map(terrain_map, inflation_radius=1):
    base_blocked = (terrain_map == ROCK) | (terrain_map == CRATER)
    blocked = inflate_obstacles(base_blocked, radius=inflation_radius)
    if hazard_blocked_map is not None:
        blocked = blocked | hazard_blocked_map
    return blocked

# =========================================================
# START/GOAL TEMİZLE
# =========================================================
def clear_zone(terrain_map, center_xy, radius=6):
    if center_xy is None:
        return
    x, y = center_xy
    cx, cy = int(round(x)), int(round(y))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r, c = cy + i, cx + j
            if 0 <= r < ROWS and 0 <= c < COLS:
                terrain_map[r, c] = FLAT

# =========================================================
# TRAVERSABILITY
# =========================================================
def is_blocked(blocked_map_local, row, col):
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return True
    return blocked_map_local[row, col]

def obstacle_clearance_cost(blocked_map_local, row, col, search_radius=6):
    if blocked_map_local is None:
        return 0.0

    key = (row, col, search_radius)
    if key in clearance_cache:
        return clearance_cache[key]

    min_dist = float("inf")
    r0 = max(0, row - search_radius)
    r1 = min(ROWS, row + search_radius + 1)
    c0 = max(0, col - search_radius)
    c1 = min(COLS, col + search_radius + 1)

    for rr in range(r0, r1):
        for cc in range(c0, c1):
            if blocked_map_local[rr, cc]:
                d = math.hypot(cc - col, rr - row)
                if d < min_dist:
                    min_dist = d

    if min_dist == float("inf"):
        value = 0.0
    elif min_dist < 2.0:
        value = 4.0
    elif min_dist < 3.0:
        value = 2.4
    elif min_dist < 5.0:
        value = 1.1
    else:
        value = 0.0

    clearance_cache[key] = value
    return value

def terrain_cost(terrain_map, slope_map_local, row, col, mode="normal", blocked_map_local=None):
    t = terrain_map[row, col]
    if t in (ROCK, CRATER):
        return None

    if t == UNKNOWN:
        base = UNKNOWN_TERRAIN_COST
        slope_val = UNKNOWN_SLOPE_ESTIMATE if np.isnan(slope_map_local[row, col]) else float(slope_map_local[row, col])
        base += slope_val * (3.6 if mode == "safe" else (3.0 if mode == "normal" else 2.4))
        if blocked_map_local is not None:
            base += obstacle_clearance_cost(blocked_map_local, row, col, search_radius=6)
        return base

    if t == ICE and ice_strength_map[row, col] > 0.76 and depth_map[row, col] > 54:
        return None
    if t == SOFT and soft_sink_map[row, col] > 0.75 and depth_map[row, col] > 49:
        return None

    base = 1.0

    if t == STEEP:
        base += 1.2 if mode != "fast" else 0.7
    elif t == ICE:
        base += 0.5 + 1.6 * float(ice_strength_map[row, col])
    elif t == SOFT:
        base += 0.5 + 1.8 * float(soft_sink_map[row, col])

    slope_factor = 4.0 if mode == "safe" else (3.0 if mode == "normal" else 2.2)
    base += float(slope_map_local[row, col]) * slope_factor

    if blocked_map_local is not None:
        base += obstacle_clearance_cost(blocked_map_local, row, col, search_radius=6)

    return base

# =========================================================
# VALIDATION / ANALYSIS
# =========================================================
def circular_cells(center_row, center_col, radius):
    cells = []
    r2 = radius * radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= r2:
                rr = center_row + dr
                cc = center_col + dc
                if 0 <= rr < ROWS and 0 <= cc < COLS:
                    cells.append((rr, cc))
    return cells

def point_metrics(x, y, terrain_map=None, slope_map_local=None, depth_map_local=None, blocked_map_local=None):
    terrain_map = terrain if terrain_map is None else terrain_map
    slope_map_local = slope_map if slope_map_local is None else slope_map_local
    depth_map_local = depth_map if depth_map_local is None else depth_map_local
    blocked_map_local = blocked_map if blocked_map_local is None else blocked_map_local

    row, col = nearest_cell(x, y)
    t = terrain_map[row, col]
    slope_val = 0.0 if np.isnan(slope_map_local[row, col]) else float(slope_map_local[row, col])

    local_cells = circular_cells(row, col, VALIDATION_RADIUS)
    unsafe_count = sum(1 for rr, cc in local_cells if blocked_map_local[rr, cc])
    unsafe_ratio = unsafe_count / max(1, len(local_cells))

    clearance = float("inf")
    for rr in range(max(0, row - CLEARANCE_SEARCH_RADIUS), min(ROWS, row + CLEARANCE_SEARCH_RADIUS + 1)):
        for cc in range(max(0, col - CLEARANCE_SEARCH_RADIUS), min(COLS, col + CLEARANCE_SEARCH_RADIUS + 1)):
            if blocked_map_local[rr, cc]:
                d = math.hypot(cc - col, rr - row)
                if d < clearance:
                    clearance = d

    if clearance == float("inf"):
        clearance = CLEARANCE_SEARCH_RADIUS + 1.0

    slip, sink, tilt, overall, warning = terrain_risk_breakdown(t, row, col)

    return {
        "row": row,
        "col": col,
        "terrain": t,
        "slope": slope_val,
        "unsafe_ratio": unsafe_ratio,
        "clearance": clearance,
        "slip": slip,
        "sink": sink,
        "tilt": tilt,
        "overall_risk": overall,
        "warning": warning,
        "depth": None if np.isnan(depth_map_local[row, col]) else float(depth_map_local[row, col]),
    }

def validate_point(x, y, point_name="POINT", terrain_map=None, slope_map_local=None, depth_map_local=None, blocked_map_local=None):
    m = point_metrics(x, y, terrain_map=terrain_map, slope_map_local=slope_map_local, depth_map_local=depth_map_local, blocked_map_local=blocked_map_local)
    reasons = []
    warnings = []
    severity = "green"
    valid = True

    blocked_map_local = blocked_map if blocked_map_local is None else blocked_map_local
    terrain_map = terrain if terrain_map is None else terrain_map
    slope_map_local = slope_map if slope_map_local is None else slope_map_local
    depth_map_local = depth_map if depth_map_local is None else depth_map_local

    if blocked_map_local[m["row"], m["col"]]:
        valid = False
        severity = "red"
        reasons.append(f"Terrain is {TERRAIN_NAMES[m['terrain']]}, not traversable.")

    if m["terrain"] == UNKNOWN:
        severity = "yellow"
        warnings.append("Point lies in unexplored area; validation is uncertain.")

    if valid and terrain_map[m["row"], m["col"]] == ICE:
        if ice_strength_map[m["row"], m["col"]] > 0.76 and depth_map[m["row"], m["col"]] > 54:
            valid = False
            severity = "red"
            reasons.append("Ice severity and local depth exceed safe entry threshold.")

    if valid and terrain_map[m["row"], m["col"]] == SOFT:
        if soft_sink_map[m["row"], m["col"]] > 0.75 and depth_map[m["row"], m["col"]] > 49:
            valid = False
            severity = "red"
            reasons.append("Sink severity and local depth exceed safe entry threshold.")

    if m["terrain"] != UNKNOWN and m["slope"] > POINT_RULES["slope_reject"]:
        valid = False
        severity = "red"
        reasons.append(
            f"Slope {m['slope']:.3f} > reject limit {POINT_RULES['slope_reject']:.2f}."
        )
    elif m["terrain"] != UNKNOWN and m["slope"] > POINT_RULES["slope_warn"]:
        severity = "yellow"
        warnings.append(
            f"Slope {m['slope']:.3f} > warning limit {POINT_RULES['slope_warn']:.2f}."
        )

    if m["unsafe_ratio"] > POINT_RULES["unsafe_ratio_reject"]:
        valid = False
        severity = "red"
        reasons.append(
            f"Unsafe ratio {m['unsafe_ratio']*100:.1f}% > reject limit {POINT_RULES['unsafe_ratio_reject']*100:.0f}%."
        )
    elif m["unsafe_ratio"] > POINT_RULES["unsafe_ratio_warn"]:
        severity = "yellow"
        warnings.append(
            f"Unsafe ratio {m['unsafe_ratio']*100:.1f}% > warning limit {POINT_RULES['unsafe_ratio_warn']*100:.0f}%."
        )

    if m["clearance"] < POINT_RULES["clearance_reject"]:
        valid = False
        severity = "red"
        reasons.append(
            f"Clearance {m['clearance']:.2f} < reject limit {POINT_RULES['clearance_reject']:.1f} cells."
        )
    elif m["clearance"] < POINT_RULES["clearance_warn"]:
        severity = "yellow"
        warnings.append(
            f"Clearance {m['clearance']:.2f} < warning limit {POINT_RULES['clearance_warn']:.1f} cells."
        )

    if valid and m["terrain"] in (ICE, SOFT, STEEP):
        severity = "yellow"
        warnings.append(f"Terrain is {TERRAIN_NAMES[m['terrain']]}, caution advised.")

    if valid and not warnings:
        summary = f"{point_name} accepted"
    elif valid:
        summary = f"{point_name} accepted with warning"
    else:
        summary = f"{point_name} rejected"

    lines = [
        summary,
        f"Terrain : {TERRAIN_NAMES[m['terrain']]}",
        f"Slope   : {m['slope']:.3f}",
        f"Unsafe  : {m['unsafe_ratio']*100:.1f}%",
        f"Clear   : {m['clearance']:.2f} cells",
        f"Risk    : {m['overall_risk']:.1f}",
    ]

    if m["terrain"] == ICE:
        lines.append(f"IceLvl  : {ice_strength_map[m['row'], m['col']]:.2f}")
    if m["terrain"] == SOFT:
        lines.append(f"SinkLvl : {soft_sink_map[m['row'], m['col']]:.2f}")
    if m['depth'] is None:
        lines.append("Depth   : unknown")
    else:
        lines.append(f"Depth   : {m['depth']:.1f}")

    if reasons:
        lines.append("Reason  : " + " | ".join(reasons))
    elif warnings:
        lines.append("Warn    : " + " | ".join(warnings))
    else:
        lines.append("Reason  : Point is valid and safe.")

    return {
        "valid": valid,
        "severity": severity,
        "metrics": m,
        "message": "\n".join(lines)
    }

def heading_step_difference(h1, h2):
    d = abs(h2 - h1)
    return min(d, NUM_HEADINGS - d)

def energy_for_transition(prev_state, next_state):
    x, y, h = next_state
    r, c = nearest_cell(x, y)
    t = terrain[r, c]
    slope_val = float(slope_map[r, c])

    turn_steps = heading_step_difference(prev_state[2], next_state[2])

    e = ENERGY_MODEL["base_step"]
    e += ENERGY_MODEL["terrain_bonus"].get(t, 0.0)
    e += ENERGY_MODEL["slope_coeff"] * slope_val

    if t == ICE:
        e += 0.10 + 0.18 * float(ice_strength_map[r, c])
    elif t == SOFT:
        e += 0.12 + 0.22 * float(soft_sink_map[r, c])

    e += ENERGY_MODEL["turn_coeff"] * turn_steps
    return e

def estimate_mission_feasibility(path, battery_level):
    if not path or len(path) < 2:
        return {
            "status": "red",
            "path_length": 0.0,
            "path_cost": float("inf"),
            "energy_required": float("inf"),
            "energy_safe": float("inf"),
            "battery": battery_level,
            "reason": "No valid route found."
        }

    path_length = 0.0
    total_cost = 0.0
    energy_required = 0.0

    for i in range(1, len(path)):
        p0 = path[i - 1]
        p1 = path[i]

        path_length += math.hypot(p1[0] - p0[0], p1[1] - p0[1])

        r, c = nearest_cell(p1[0], p1[1])
        move_cost = terrain_cost(terrain, slope_map, r, c, mode="normal", blocked_map_local=blocked_map)
        if move_cost is None:
            move_cost = 999.0
        total_cost += move_cost
        energy_required += energy_for_transition(p0, p1)

    energy_safe = energy_required * (1.0 + ENERGY_MODEL["reserve_margin"])

    if battery_level >= energy_safe:
        status = "green"
        reason = "Battery is sufficient with safety reserve."
    elif battery_level >= energy_required:
        status = "yellow"
        reason = "Mission is possible, but reserve is low. Recharge advised."
    else:
        status = "red"
        reason = "Battery is insufficient. Recharge required before mission."

    return {
        "status": status,
        "path_length": path_length,
        "path_cost": total_cost,
        "energy_required": energy_required,
        "energy_safe": energy_safe,
        "battery": battery_level,
        "reason": reason,
    }

def mission_report_text(rep):
    return (
        "MISSION ANALYSIS\n"
        f"Length   : {rep['path_length']:.1f} m\n"
        f"Cost     : {rep['path_cost']:.1f}\n"
        f"Need     : {rep['energy_required']:.1f}%\n"
        f"Reserve  : {rep['energy_safe']:.1f}%\n"
        f"Battery  : {rep['battery']:.1f}%\n"
        f"Status   : {rep['status'].upper()}\n"
        f"Reason   : {rep['reason']}"
    )

# =========================================================
# PLANLAYICI
# =========================================================
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def plan_path_hybrid(terrain_map, slope_map_local, blocked_map_local, start_xy, goal_xy, mode="normal"):
    sx, sy = start_xy
    gx, gy = goal_xy

    angle_to_goal = math.degrees(math.atan2(gy - sy, gx - sx))
    if angle_to_goal < 0:
        angle_to_goal += 360
    start_heading = int(round(angle_to_goal / TURN_ANGLE)) % NUM_HEADINGS

    start_state = (round(sx, 1), round(sy, 1), start_heading)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_state))

    came_from = {}
    g_score = {start_state: 0.0}
    visited = set()

    start_time = time.perf_counter()
    TIME_LIMIT = 5.5
    MAX_EXPANSIONS = 65000
    expansions = 0

    while open_heap:
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            return []

        if time.perf_counter() - start_time > TIME_LIMIT:
            return []

        _, current = heapq.heappop(open_heap)
        cx, cy, ch = current

        current_key = (round(cx, 1), round(cy, 1), ch)
        if current_key in visited:
            continue
        visited.add(current_key)

        if heuristic(cx, cy, gx, gy) < 2.0:
            return reconstruct_path(came_from, current)

        for dh in (-1, 0, 1):
            nh = wrap_heading(ch + dh)
            deg = heading_to_deg(nh)
            rad = deg_to_rad(deg)

            nx = cx + STEP_LENGTH * math.cos(rad)
            ny = cy + STEP_LENGTH * math.sin(rad)

            if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 1:
                continue

            row, col = nearest_cell(nx, ny)
            if is_blocked(blocked_map_local, row, col):
                continue

            move_cost = terrain_cost(
                terrain_map,
                slope_map_local,
                row,
                col,
                mode=mode,
                blocked_map_local=blocked_map_local
            )
            if move_cost is None:
                continue

            turn_penalty = 0.32 if dh != 0 else 0.0
            if mode == "safe" and dh != 0:
                turn_penalty += 0.14

            new_g = g_score[current] + move_cost + turn_penalty
            neighbor = (round(nx, 1), round(ny, 1), nh)

            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                h = heuristic(nx, ny, gx, gy)

                if mode == "safe":
                    f = new_g + h * 0.95
                elif mode == "fast":
                    f = new_g + h * 1.05
                else:
                    f = new_g + h

                heapq.heappush(open_heap, (f, neighbor))
                came_from[neighbor] = current

    return []

def plan_path_fallback(terrain_map, slope_map_local, blocked_map_local, start_xy, goal_xy):
    coarse_step = 1.8
    coarse_headings = 12
    coarse_turn_angle = 360 / coarse_headings

    sx, sy = start_xy
    gx, gy = goal_xy

    angle_to_goal = math.degrees(math.atan2(gy - sy, gx - sx))
    if angle_to_goal < 0:
        angle_to_goal += 360
    start_heading = int(round(angle_to_goal / coarse_turn_angle)) % coarse_headings

    start_state = (int(round(sx)), int(round(sy)), start_heading)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_state))

    came_from = {}
    g_score = {start_state: 0.0}
    visited = set()

    start_time = time.perf_counter()
    TIME_LIMIT = 2.0
    MAX_EXPANSIONS = 25000
    expansions = 0

    while open_heap:
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            return []

        if time.perf_counter() - start_time > TIME_LIMIT:
            return []

        _, current = heapq.heappop(open_heap)
        cx, cy, ch = current

        if current in visited:
            continue
        visited.add(current)

        if heuristic(cx, cy, gx, gy) < 3.0:
            return reconstruct_path(came_from, current)

        for dh in (-1, 0, 1):
            nh = (ch + dh) % coarse_headings
            deg = nh * coarse_turn_angle
            rad = math.radians(deg)

            nx = int(round(cx + coarse_step * math.cos(rad)))
            ny = int(round(cy + coarse_step * math.sin(rad)))

            if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 1:
                continue

            if is_blocked(blocked_map_local, ny, nx):
                continue

            move_cost = terrain_cost(
                terrain_map,
                slope_map_local,
                ny,
                nx,
                mode="safe",
                blocked_map_local=blocked_map_local
            )
            if move_cost is None:
                continue

            new_g = g_score[current] + move_cost + (0.22 if dh != 0 else 0.0)
            neighbor = (nx, ny, nh)

            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                h = heuristic(nx, ny, gx, gy)
                f = new_g + 1.25 * h
                heapq.heappush(open_heap, (f, neighbor))
                came_from[neighbor] = current

    return []

# =========================================================
# GÖRSEL AY FOTOĞRAFI
# =========================================================
def make_lunar_rgb(heightmap, crater_bowl=None, crater_rim=None):
    ls = LightSource(azdeg=315, altdeg=38)
    rgb = ls.shade(
        heightmap,
        cmap=plt.cm.gist_gray,
        vert_exag=8.5,
        blend_mode='soft'
    )

    rgb[..., 0] *= 0.95
    rgb[..., 1] *= 0.95
    rgb[..., 2] *= 1.02

    if crater_bowl is not None and crater_rim is not None:
        bowl_norm = crater_bowl / (np.max(crater_bowl) + 1e-9)
        rim_norm = crater_rim / (np.max(crater_rim) + 1e-9)
        yy, xx = np.mgrid[0:heightmap.shape[0], 0:heightmap.shape[1]]
        light_dx, light_dy = -0.72, -0.58
        directional = ((xx / max(1, heightmap.shape[1] - 1)) * light_dx +
                       (yy / max(1, heightmap.shape[0] - 1)) * light_dy)
        directional = (directional - directional.min()) / (directional.max() - directional.min() + 1e-9)

        bowl_shadow = np.clip(0.55 + 0.45 * (1.0 - directional), 0.0, 1.0) * bowl_norm
        rim_highlight = np.clip(0.45 + 0.55 * directional, 0.0, 1.0) * rim_norm

        rgb *= (1.0 - 0.32 * bowl_shadow[..., None])
        rgb += 0.16 * rim_highlight[..., None]
        rgb -= 0.05 * bowl_norm[..., None]

    return np.clip(rgb, 0, 1)

# =========================================================
# HARİTA ÜRET
# =========================================================
heightmap, crater_mask, rock_mask, ice_mask, soft_mask, ice_strength_map, soft_sink_map, crater_bowl_map, crater_rim_map = generate_lunar_heightmap(ROWS, COLS)
terrain, slope_map = build_terrain(heightmap, crater_mask, rock_mask, ice_mask, soft_mask)

depth_map = (1.0 - heightmap) ** 1.35 * 100.0

hazard_blocked_map = (
    ((terrain == ICE) & (ice_strength_map > 0.76) & (depth_map > 54)) |
    ((terrain == SOFT) & (soft_sink_map > 0.75) & (depth_map > 49))
)

blocked_map = build_blocked_map(terrain, inflation_radius=OBSTACLE_INFLATION_RADIUS)
initialize_known_maps()
lunar_rgb = make_lunar_rgb(heightmap, crater_bowl_map, crater_rim_map)
orbital_rgb = make_orbital_rgb(lunar_rgb, terrain, depth_map)

# =========================================================
# FİGÜR / EKSENLER
# =========================================================
fig = plt.figure(figsize=(16, 10), facecolor="#f2f2f2")

ax = fig.add_axes([0.04, 0.10, 0.52, 0.82])
depth_cbar_ax = fig.add_axes([0.545, 0.15, 0.012, 0.65])
dashboard_ax = fig.add_axes([0.60, 0.05, 0.38, 0.90])
lidar_ax = fig.add_axes([0.79, 0.30, 0.17, 0.18], projection="3d")
orbital_ax = fig.add_axes([0.82, 0.08, 0.14, 0.18])

dashboard_ax.set_xlim(0, 1)
dashboard_ax.set_ylim(0, 1)
dashboard_ax.axis("off")
dashboard_ax.set_facecolor("#0f1320")
orbital_ax.set_facecolor("#0b1733")

# =========================================================
# HARİTA ÇİZİMİ
# =========================================================
ax.imshow(
    lunar_rgb,
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest"
)

terrain_overlay = np.zeros((ROWS, COLS, 4), dtype=float)
terrain_overlay[terrain == ROCK]   = [0.95, 0.95, 0.95, 0.42]
terrain_overlay[terrain == CRATER] = [0.10, 0.10, 0.10, 0.26]
terrain_overlay[terrain == STEEP]  = [0.95, 0.85, 0.25, 0.20]
terrain_overlay[terrain == ICE]    = [0.43, 0.77, 1.00, 0.20]
terrain_overlay[terrain == SOFT]   = [0.72, 0.52, 0.35, 0.22]

terrain_overlay[hazard_blocked_map & (terrain == ICE)]  = [0.10, 0.55, 1.00, 0.42]
terrain_overlay[hazard_blocked_map & (terrain == SOFT)] = [0.55, 0.22, 0.10, 0.40]

ax.imshow(
    terrain_overlay,
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest"
)

crater_floor_overlay = np.zeros((ROWS, COLS, 4), dtype=float)
crater_floor_overlay[..., 0] = 0.02
crater_floor_overlay[..., 1] = 0.02
crater_floor_overlay[..., 2] = 0.02
crater_floor_overlay[..., 3] = np.clip(0.28 * crater_bowl_map, 0.0, 0.30)

crater_rim_overlay = np.zeros((ROWS, COLS, 4), dtype=float)
crater_rim_overlay[..., 0] = 0.96
crater_rim_overlay[..., 1] = 0.95
crater_rim_overlay[..., 2] = 0.88
crater_rim_overlay[..., 3] = np.clip(0.24 * crater_rim_map, 0.0, 0.24)

ax.imshow(
    crater_floor_overlay,
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest"
)

ax.imshow(
    crater_rim_overlay,
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest"
)

ax.imshow(
    depth_map,
    extent=[0, COLS, ROWS, 0],
    cmap="Greys",
    alpha=0.20,
    interpolation="nearest",
    vmin=0,
    vmax=100
)

crater_depth_masked = np.ma.masked_where(~crater_mask, depth_map)
crater_levels = [36, 52, 68, 82]
ax.contour(
    np.arange(COLS),
    np.arange(ROWS),
    crater_depth_masked,
    levels=crater_levels,
    colors=["#8e8e8e", "#6d6d6d", "#555555", "#3f3f3f"],
    linewidths=[0.35, 0.42, 0.50, 0.58],
    alpha=0.62
)

fog_overlay_img = ax.imshow(
    np.ones((ROWS, COLS, 4), dtype=float),
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest",
    zorder=5
)

sensor_ring = Circle((0, 0), SENSOR_RADIUS, fill=False, edgecolor="#ffffff", linewidth=1.4, alpha=0.85, zorder=9)
sensor_ring.set_visible(False)
ax.add_patch(sensor_ring)

planning_overlay_img = ax.imshow(
    orbital_rgb,
    extent=[0, COLS, ROWS, 0],
    interpolation="nearest",
    zorder=10,
    alpha=0.98
)

orbital_ax.imshow(orbital_rgb, extent=[0, COLS, ROWS, 0], interpolation="nearest")
orbital_ax.set_xticks([])
orbital_ax.set_yticks([])
for spine in orbital_ax.spines.values():
    spine.set_edgecolor("white")
    spine.set_linewidth(1.0)
orbital_start_marker, = orbital_ax.plot([], [], marker="o", markersize=6, color="deepskyblue", linestyle="None")
orbital_goal_marker, = orbital_ax.plot([], [], marker="X", markersize=7, color="#ff5252", linestyle="None")
orbital_rover_marker, = orbital_ax.plot([], [], marker="^", markersize=7, color="orange", linestyle="None")
orbital_trail_line, = orbital_ax.plot([], [], color="#ffb347", linewidth=1.0, alpha=0.9)

update_lidar_panel()

ax.set_title("Ay Rover Simulasyonu", fontsize=24, pad=12)
ax.set_xlabel("X (metre)", fontsize=12)
ax.set_ylabel("Y (metre)", fontsize=12)
ax.set_xticks(np.arange(0, COLS + 1, 10))
ax.set_yticks(np.arange(0, ROWS + 1, 10))
ax.grid(color="white", linestyle="--", linewidth=0.25, alpha=0.18)

# Haritayi eksene tam oturt
ax.set_xlim(0, COLS)
ax.set_ylim(ROWS, 0)
ax.margins(0, 0)
ax.set_aspect("equal", adjustable="box")

# =========================================================
# DEPTH COLORBAR
# =========================================================
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap="Greys", norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=depth_cbar_ax)
cbar.set_ticks([0, 25, 50, 75, 100])
cbar.ax.tick_params(colors="black", labelsize=9)
plt.setp(cbar.ax.get_yticklabels(), color="black")
cbar.outline.set_edgecolor("black")
cbar.set_label("Depth Level (Dark = Deeper)", color="black", fontsize=10, labelpad=10)
depth_cbar_ax.set_facecolor("white")

fig.text(0.591, 0.82, "HIGH", color="black", fontsize=9, ha="center", fontweight="bold")
fig.text(0.591, 0.12, "LOW", color="black", fontsize=9, ha="center", fontweight="bold")

start_scatter = None
goal_scatter = None

rover_marker, = ax.plot([], [], marker="^", markersize=15, color="orange", linestyle="None", zorder=8)
path_line, = ax.plot([], [], color="yellow", linewidth=2.0, alpha=0.95, zorder=6)
trail_line, = ax.plot([], [], color="#ff9933", linewidth=1.6, alpha=0.85, zorder=7)
opt1_line, = ax.plot([], [], color="#66e0ff", linewidth=2.2, linestyle="--", alpha=0.95, zorder=7)
opt2_line, = ax.plot([], [], color="#ff66d9", linewidth=2.2, linestyle="--", alpha=0.95, zorder=7)

map_info = ax.text(
    3, 7,
    "Haritaya tikla: START sec\nBilinmeyen alanlar soluk gorunur.",
    fontsize=11,
    color="white",
    bbox=dict(facecolor="black", alpha=0.70, boxstyle="round,pad=0.4"),
    zorder=30
)

# =========================================================
# DASHBOARD STİL
# =========================================================
SMALL = 8.0
MED = 9.0
BIG = 10.8

CARD_BG = "#07122b"
CARD_EDGE = "#253553"
CARD_INNER = CARD_BG
CARD_HILITE = "#31456f"
TEXT_MAIN = "white"
TEXT_MUTED = "#cfd6e6"
ACCENT = "#8ad7ff"

outer_panel = Rectangle(
    (0.01, 0.02), 0.98, 0.96,
    facecolor=CARD_BG,
    edgecolor=CARD_EDGE,
    lw=1.0
)
dashboard_ax.add_patch(outer_panel)

def draw_section(x, y, w, h, title, subtitle=None):
    dashboard_ax.text(
        x + 0.025, y + h - 0.020,
        title,
        color=TEXT_MAIN,
        fontsize=BIG,
        fontweight="bold",
        va="top"
    )
    if subtitle:
        dashboard_ax.text(
            x + 0.025, y + h - 0.058,
            subtitle,
            color=ACCENT,
            fontsize=SMALL - 0.5,
            va="top",
            fontweight="bold"
        )

status_sec   = (0.02, 0.81, 0.46, 0.17)
terrain_sec  = (0.52, 0.81, 0.46, 0.17)
decision_sec = (0.02, 0.63, 0.96, 0.15)
legend_sec   = (0.02, 0.28, 0.64, 0.31)
mission_sec  = (0.70, 0.28, 0.28, 0.31)
sensors_sec  = (0.02, 0.05, 0.96, 0.18)

draw_section(*status_sec, "ROVER STATUS")
draw_section(*terrain_sec, "TERRAIN ANALYSIS")
draw_section(*decision_sec, "DECISION ENGINE")
draw_section(*legend_sec, "LEGEND")
draw_section(*mission_sec, "MISSION LOG")
draw_section(*sensors_sec, "ACTIVE SENSORS", "Sensor Fusion Feed")

status_text = dashboard_ax.text(
    0.06, 0.905, "",
    color=TEXT_MAIN,
    fontsize=SMALL - 0.9,
    va="top",
    family="monospace",
    linespacing=1.14
)

dashboard_ax.text(0.08, 0.800, "Energy", color=TEXT_MAIN, fontsize=SMALL - 0.5)
dashboard_ax.add_patch(Rectangle((0.08, 0.782), 0.32, 0.022, facecolor="#2b2f3a", edgecolor="none"))
energy_bar = Rectangle((0.08, 0.782), 0.32, 0.022, facecolor="#4bd36b", edgecolor="none")
dashboard_ax.add_patch(energy_bar)

terrain_text = dashboard_ax.text(
    0.55, 0.905, "",
    color=TEXT_MAIN,
    fontsize=SMALL - 0.9,
    va="top",
    family="monospace",
    linespacing=1.14
)

dashboard_ax.text(0.57, 0.800, "Risk", color=TEXT_MAIN, fontsize=SMALL - 0.5)
dashboard_ax.add_patch(Rectangle((0.57, 0.782), 0.32, 0.022, facecolor="#2b2f3a", edgecolor="none"))
risk_bar = Rectangle((0.57, 0.782), 0.32, 0.022, facecolor="#ffd84d", edgecolor="none")
dashboard_ax.add_patch(risk_bar)

decision_text = dashboard_ax.text(
    0.06, 0.695, "",
    color=TEXT_MAIN,
    fontsize=MED - 0.4,
    va="top",
    family="monospace",
    linespacing=1.24
)

legend_rows = {}
legend_start_y = 0.525
legend_gap = 0.038

for i, t in enumerate([FLAT, ICE, SOFT, STEEP, ROCK, CRATER, UNKNOWN]):
    y = legend_start_y - i * legend_gap
    bg = None
    swatch = Rectangle((0.08, y - 0.009), 0.022, 0.018, facecolor=TERRAIN_COLORS[t], edgecolor="none")
    txt = dashboard_ax.text(
        0.12, y,
        f"{TERRAIN_NAMES[t]:<7} -> {legend_desc[t]}",
        color=TEXT_MUTED,
        fontsize=SMALL - 0.5,
        va="center",
        family="monospace"
    )
    dashboard_ax.add_patch(swatch)
    legend_rows[t] = (bg, txt)

log_lines = []
log_text_objs = [
    dashboard_ax.text(
        0.715,
        0.555 - i * 0.056,
        "",
        color="#d8dde8",
        fontsize=7.2,
        va="top",
        family="monospace"
    )
    for i in range(5)
]

def shorten_log(msg, max_len=18):
    return msg if len(msg) <= max_len else msg[:max_len - 3] + "..."

def push_log(msg):
    log_lines.append(shorten_log(msg))
    if len(log_lines) > 5:
        log_lines.pop(0)
    for i, t in enumerate(log_text_objs):
        idx = len(log_lines) - 1 - i
        t.set_text(log_lines[idx] if idx >= 0 else "")

sensor_text_objs = [
    dashboard_ax.text(
        0.05,
        0.145 - i * 0.032,
        "",
        color=TEXT_MUTED,
        fontsize=8.2,
        va="top",
        family="monospace",
        clip_on=True
    )
    for i in range(4)
]

def active_sensor_lines(terrain_type, row=None, col=None):
    if terrain_type is None or terrain_type == UNKNOWN or row is None or col is None:
        return [
            ("LIDAR", "Local ranging standby", "#d8dde8"),
            ("StereoCam", "Awaiting terrain lock", "#d8dde8"),
            ("IMU", "Idle stability monitor", "#d8dde8"),
            ("WheelSense", "Traction baseline only", "#d8dde8"),
        ]

    lines = [
        ("LIDAR", "Forward obstacle scan", "#8ad7ff"),
        ("StereoCam", "Local texture classify", "#d8dde8"),
        ("IMU", "Body stability monitor", "#d8dde8"),
        ("WheelSense", "Traction / slip feedback", "#d8dde8"),
    ]

    if terrain_type == ROCK:
        lines = [
            ("LIDAR", "Rock obstacle locked", "#ff8f8f"),
            ("StereoCam", "Rock texture confirm", "#ffd84d"),
            ("IMU", "Bypass posture hold", "#d8dde8"),
            ("WheelSense", "Obstacle stop check", "#ff8f8f"),
        ]
    elif terrain_type == CRATER:
        depth_level = float(depth_map[row, col])
        color = "#ff8f8f" if depth_level > 60 else "#ffd84d"
        lines = [
            ("LIDAR", "Crater rim profile", "#8ad7ff"),
            ("StereoCam", "Rim / shadow classify", "#ffd84d"),
            ("IMU", "Tilt hazard escalation", color),
            ("WheelSense", "Entry denied / sink check", "#ff8f8f"),
        ]
    elif terrain_type == STEEP:
        slope_val = float(slope_map[row, col])
        color = "#ff8f8f" if slope_val > POINT_RULES["slope_warn"] else "#ffd84d"
        lines = [
            ("LIDAR", "Surface profile tracing", "#8ad7ff"),
            ("StereoCam", "Gradient texture support", "#d8dde8"),
            ("IMU", f"Tilt high ({slope_val:.3f})", color),
            ("WheelSense", "Grip compensation active", "#ffd84d"),
        ]
    elif terrain_type == ICE:
        ice_level = float(ice_strength_map[row, col])
        color = "#ff8f8f" if ice_level > 0.76 else "#ffd84d"
        lines = [
            ("LIDAR", "Surface smoothness scan", "#8ad7ff"),
            ("StereoCam", f"Ice patch prob {ice_level:.2f}", color),
            ("IMU", "Slip-sensitive heading hold", "#ffd84d"),
            ("WheelSense", "Traction loss monitor", color),
        ]
    elif terrain_type == SOFT:
        sink_level = float(soft_sink_map[row, col])
        color = "#ff8f8f" if sink_level > 0.75 else "#ffd84d"
        lines = [
            ("LIDAR", "Surface depression scan", "#8ad7ff"),
            ("StereoCam", f"Soft soil prob {sink_level:.2f}", color),
            ("IMU", "Body settling monitor", "#ffd84d"),
            ("WheelSense", "Sink / slip feedback", color),
        ]
    elif terrain_type == FLAT:
        lines = [
            ("LIDAR", "Forward clearance nominal", "#8ad7ff"),
            ("StereoCam", "Flat terrain confirmed", "#4bd36b"),
            ("IMU", "Stable chassis state", "#4bd36b"),
            ("WheelSense", "Traction nominal", "#4bd36b"),
        ]

    return lines

def update_sensor_panel(terrain_type, row=None, col=None):
    lines = active_sensor_lines(terrain_type, row, col)
    for txt_obj, (name, desc, color) in zip(sensor_text_objs, lines):
        txt_obj.set_text(f"{name:<11}: {desc}")
        txt_obj.set_color(color)

# =========================================================
# SİMÜLASYON DURUMU
# =========================================================
selection_stage = 0
sim_state = "idle"
energy = 100.0
current_risk = 0.0

path_states = []
path_x = []
path_y = []

current_path = []
current_index = 0
trail_x = []
trail_y = []

decision_option_safe = []
decision_option_fast = []
chosen_option = None
resume_main_index = None

timer = None
last_planner_mode = "normal"

# =========================================================
# DASHBOARD GÜNCELLEME
# =========================================================
def update_live_legend(active_terrain):
    for t, (bg, txt) in legend_rows.items():
        if t == active_terrain:
            txt.set_color("white")
            txt.set_fontweight("bold")
        else:
            txt.set_color(TEXT_MUTED)
            txt.set_fontweight("normal")

def update_dashboard(x=None, y=None, heading_idx=None, terrain_type=None, speed=0.0,
                     decision="Waiting", reason="Awaiting command"):
    global current_risk, energy

    rr, cc = nearest_cell(x, y) if x is not None and y is not None else (None, None)

    if terrain_type is None:
        terrain_name = "NONE"
        slip = sink = tilt = overall = 0
        warning = "Select START and GOAL"
        update_live_legend(None)
        current_risk = 0.0
        extra_line = ""
        update_sensor_panel(None, None, None)
    else:
        terrain_name = TERRAIN_NAMES[terrain_type]
        update_live_legend(terrain_type)

        if terrain_type == UNKNOWN:
            slip = sink = tilt = 0
            overall = 18
            warning = "Unexplored terrain - sensor scan required"
            current_risk = overall
            extra_line = "\nDepth        : unknown"
        else:
            slip, sink, tilt, overall, warning = terrain_risk_breakdown(terrain_type, rr, cc)
            current_risk = overall
            if terrain_type == ICE:
                extra_line = (
                    f"\nIce Sev      : {ice_strength_map[rr, cc]:.2f}"
                    f"\nDepth        : {depth_map[rr, cc]:.1f}"
                )
            elif terrain_type == SOFT:
                extra_line = (
                    f"\nSink Sev     : {soft_sink_map[rr, cc]:.2f}"
                    f"\nDepth        : {depth_map[rr, cc]:.1f}"
                )
            else:
                extra_line = f"\nDepth        : {depth_map[rr, cc]:.1f}"

        update_sensor_panel(terrain_type, rr, cc)

    pos_str = "(-, -)" if x is None else f"({x:.1f}, {y:.1f})"
    head_str = "-" if heading_idx is None else f"{heading_to_deg(heading_idx):.0f}°"

    status_text.set_text(
        f"Position : {pos_str}\n"
        f"Heading  : {head_str}\n"
        f"Speed    : {speed:.2f} m/s\n"
        f"Energy   : {energy:.1f}%\n"
        f"State    : {sim_state.upper()}"
    )

    terrain_text.set_text(
        f"Terrain      : {terrain_name}\n"
        f"Slip Risk    : {slip}%\n"
        f"Sink Risk    : {sink}%\n"
        f"Tilt Risk    : {tilt}%\n"
        f"Overall Risk : {overall}%"
        f"{extra_line}"
    )

    decision_text.set_text(
        f"Decision : {decision}\n"
        f"Reason   : {reason}\n"
        f"Warning  : {warning}"
    )

    update_lidar_panel(x, y, heading_idx)

    energy_bar.set_width(0.28 * max(0.0, min(1.0, energy / 100.0)))
    if energy > 60:
        energy_bar.set_facecolor("#4bd36b")
    elif energy > 30:
        energy_bar.set_facecolor("#ffd84d")
    else:
        energy_bar.set_facecolor("#ff6666")

    risk_bar.set_width(0.28 * max(0.0, min(1.0, current_risk / 100.0)))
    if current_risk < 35:
        risk_bar.set_facecolor("#4bd36b")
    elif current_risk < 70:
        risk_bar.set_facecolor("#ffd84d")
    else:
        risk_bar.set_facecolor("#ff6666")

update_dashboard()
update_lidar_panel()
set_main_view("planning")
refresh_visibility_overlay()

# =========================================================
# REPLAN / RETREAT YARDIMCILARI
# =========================================================
def register_safe_pose(x, y, heading_idx):
    global safe_history

    r, c = nearest_cell(x, y)

    if known_blocked_map is not None and known_blocked_map[r, c]:
        return

    terrain_local = planner_terrain_map()
    slope_local = planner_slope_map()

    move_cost = terrain_cost(
        terrain_local,
        slope_local,
        r,
        c,
        mode="safe",
        blocked_map_local=known_blocked_map
    )
    if move_cost is None:
        return

    state = (round(x, 1), round(y, 1), heading_idx)

    if safe_history:
        px, py, _ = safe_history[-1]
        if math.hypot(x - px, y - py) < 1.5:
            safe_history[-1] = state
            return

    safe_history.append(state)

    if len(safe_history) > MAX_RETREAT_POINTS:
        safe_history = safe_history[-MAX_RETREAT_POINTS:]

def build_retreat_path(current_state):
    if not safe_history:
        return []

    terrain_local = planner_terrain_map()
    slope_local = planner_slope_map()
    blocked_local = known_blocked_map

    if blocked_local is None:
        return []

    candidates = list(reversed(safe_history[:-1])) if len(safe_history) > 1 else []

    checked = 0
    for anchor in candidates:
        checked += 1
        if checked > 12:
            break

        if math.hypot(current_state[0] - anchor[0], current_state[1] - anchor[1]) < RETREAT_MIN_ANCHOR_GAP:
            continue

        seg = plan_path_hybrid(
            terrain_local,
            slope_local,
            blocked_local,
            (current_state[0], current_state[1]),
            (anchor[0], anchor[1]),
            mode="safe"
        )

        if len(seg) <= 2:
            seg = plan_path_fallback(
                terrain_local,
                slope_local,
                blocked_local,
                (current_state[0], current_state[1]),
                (anchor[0], anchor[1])
            )

        if len(seg) > 2:
            return seg

    return []

def should_attempt_replan(current_path_local, idx, x, y):
    global last_replan_fail_time, last_replan_fail_cell

    if GOAL is None or current_path_local is not path_states:
        return False

    if idx < REPLAN_MIN_PROGRESS_STEPS:
        return False

    now = time.perf_counter()
    if now - last_replan_time < REPLAN_COOLDOWN_SEC:
        return False

    cell = nearest_cell(x, y)
    if last_replan_fail_cell == cell and (now - last_replan_fail_time) < REPLAN_FAIL_IGNORE_SEC:
        return False

    return True

# =========================================================
# YOL HESAPLAMA
# =========================================================
def recompute_main_path(start_xy=None):
    global path_states, path_x, path_y, mission_report, blocked_map, last_planner_mode

    if GOAL is None:
        return False

    plan_start = START if start_xy is None else start_xy

    clearance_cache.clear()

    terrain_local = planner_terrain_map()
    slope_local = planner_slope_map()

    clear_zone(terrain_local, plan_start, radius=4)
    clear_zone(terrain_local, GOAL, radius=3)

    blocked_local = known_blocked_map.copy() if known_blocked_map is not None else build_blocked_map(terrain_local, inflation_radius=OBSTACLE_INFLATION_RADIUS)

    start_plan = time.perf_counter()
    new_path = plan_path_hybrid(terrain_local, slope_local, blocked_local, plan_start, GOAL, mode="normal")
    plan_elapsed = time.perf_counter() - start_plan

    if len(new_path) <= 2:
        clearance_cache.clear()
        fallback_start = time.perf_counter()
        new_path = plan_path_fallback(terrain_local, slope_local, blocked_local, plan_start, GOAL)
        fallback_elapsed = time.perf_counter() - fallback_start
        last_planner_mode = f"fallback ({fallback_elapsed:.2f}s)"
    else:
        last_planner_mode = f"hybrid ({plan_elapsed:.2f}s)"

    if len(new_path) <= 2:
        mission_report = None
        return False

    blocked_map = build_blocked_map(terrain, inflation_radius=OBSTACLE_INFLATION_RADIUS)
    path_states = new_path
    path_x = [s[0] for s in path_states]
    path_y = [s[1] for s in path_states]

    mission_report = estimate_mission_feasibility(path_states, energy)
    return True

def path_state_is_invalid(state):
    x, y, _ = state
    r, c = nearest_cell(x, y)

    if known_blocked_map is not None and known_blocked_map[r, c]:
        return True, "Known obstacle on route"

    terrain_local = planner_terrain_map()
    slope_local = planner_slope_map()
    move_cost = terrain_cost(terrain_local, slope_local, r, c, mode="normal", blocked_map_local=known_blocked_map)
    if move_cost is None:
        return True, "Route segment became unsafe"

    return False, ""

def upcoming_path_hazard(path_local, start_idx=0, lookahead=REPLAN_LOOKAHEAD):
    if not path_local:
        return True, "No route available"

    end_idx = min(len(path_local), start_idx + lookahead)
    for idx in range(start_idx, end_idx):
        invalid, reason = path_state_is_invalid(path_local[idx])
        if invalid:
            return True, reason
    return False, ""

def replan_from_pose(x, y, heading_idx, trigger_reason="New hazard detected"):
    global current_path, current_index, motion_substep, sim_state, last_replan_time
    global last_replan_fail_time, last_replan_fail_cell
    global path_states, path_x, path_y

    now = time.perf_counter()
    if now - last_replan_time < REPLAN_COOLDOWN_SEC:
        return "cooldown"

    current_state = (round(x, 1), round(y, 1), heading_idx)

    ok = recompute_main_path(start_xy=(x, y))
    last_replan_time = now

    if ok:
        current_path = path_states
        current_index = 0
        motion_substep = 0
        clear_decision_visuals()
        path_line.set_data(path_x, path_y)
        sim_state = "moving"

        push_log("Route replanned")
        push_log(last_planner_mode)
        map_info.set_text(
            "ONLINE REPLANNING\n"
            f"Reason: {trigger_reason}\n"
            "Rover yeni gozlemlere gore rotayi guncelledi."
        )
        update_dashboard(
            x=x,
            y=y,
            heading_idx=heading_idx,
            terrain_type=terrain_label_for_display(*nearest_cell(x, y)),
            speed=0.0,
            decision="Replanning",
            reason=trigger_reason
        )
        return "replanned"

    retreat_path = build_retreat_path(current_state)
    if retreat_path:
        retreat_end = retreat_path[-1]
        anchor_xy = (retreat_end[0], retreat_end[1])
        ok2 = recompute_main_path(start_xy=anchor_xy)

        if ok2:
            current_path = retreat_path
            current_index = 0
            motion_substep = 0
            clear_decision_visuals()
            sim_state = "moving"

            push_log("Dead-end retreat")
            push_log(last_planner_mode)
            map_info.set_text(
                "RETREAT + REPLAN\n"
                f"Reason: {trigger_reason}\n"
                "Rover cikmaz algiladi, guvenli noktaya geri cekiliyor."
            )
            update_dashboard(
                x=x,
                y=y,
                heading_idx=heading_idx,
                terrain_type=terrain_label_for_display(*nearest_cell(x, y)),
                speed=0.0,
                decision="Retreat",
                reason="Dead-end escape"
            )
            return "retreat"

    last_replan_fail_time = now
    last_replan_fail_cell = nearest_cell(x, y)

    push_log("Replan failed")
    update_dashboard(
        x=x,
        y=y,
        heading_idx=heading_idx,
        terrain_type=terrain_label_for_display(*nearest_cell(x, y)),
        speed=0.0,
        decision="Continue",
        reason=f"{trigger_reason} / keep current motion"
    )
    map_info.set_text(
        "REPLAN FAILED\n"
        "Alternatif rota hemen bulunamadi.\n"
        "Rover mevcut guvenli animasyon adimini tamamlayip tekrar deneyecek."
    )
    sim_state = "moving"
    return "failed"

# =========================================================
# KARAR SİSTEMİ
# =========================================================
def local_path_segment(start_state, goal_state, mode):
    sxy = (start_state[0], start_state[1])
    gxy = (goal_state[0], goal_state[1])

    seg = plan_path_hybrid(planner_terrain_map(), planner_slope_map(), known_blocked_map, sxy, gxy, mode=mode)
    if not seg:
        seg = plan_path_fallback(planner_terrain_map(), planner_slope_map(), known_blocked_map, sxy, gxy)
    return seg

def score_segment(seg):
    if not seg:
        return float("inf"), float("inf")

    total_cost = 0.0
    total_risk = 0.0
    for state in seg:
        x, y, _ = state
        r, c = nearest_cell(x, y)
        t = known_terrain[r, c] if explored_mask[r, c] else UNKNOWN

        move_c = terrain_cost(planner_terrain_map(), planner_slope_map(), r, c, mode="normal", blocked_map_local=known_blocked_map)
        if move_c is None:
            return float("inf"), float("inf")

        total_cost += move_c

        slip, sink, tilt, overall, _ = terrain_risk_breakdown(t, r, c)
        total_risk += overall * 0.18
        total_risk += obstacle_clearance_cost(known_blocked_map, r, c, search_radius=6) * 8.0

    return total_cost, total_risk

def show_decision_options():
    global decision_option_safe, decision_option_fast, chosen_option, resume_main_index, sim_state

    if current_index <= 0 or current_index >= len(path_states) - 3:
        return False

    anchor_idx = min(len(path_states) - 1, current_index + DECISION_LOOKAHEAD)
    current_state = current_path[current_index]
    anchor_state = path_states[anchor_idx]

    safe_seg = local_path_segment(current_state, anchor_state, mode="safe")
    fast_seg = local_path_segment(current_state, anchor_state, mode="fast")

    if not safe_seg or not fast_seg:
        return False

    decision_option_safe = safe_seg
    decision_option_fast = fast_seg

    safe_cost, safe_risk = score_segment(safe_seg)
    fast_cost, fast_risk = score_segment(fast_seg)

    score_safe = safe_cost + safe_risk * 1.1
    score_fast = fast_cost + fast_risk * 0.8

    if safe_risk <= fast_risk * 1.15:
        chosen_option = "safe"
    else:
        chosen_option = "fast" if score_fast < score_safe else "safe"

    resume_main_index = anchor_idx

    opt1_line.set_data([p[0] for p in safe_seg], [p[1] for p in safe_seg])
    opt2_line.set_data([p[0] for p in fast_seg], [p[1] for p in fast_seg])

    reason = "Safety prioritized over shorter route" if chosen_option == "safe" else "Fast route accepted within risk tolerance"
    cur_r, cur_c = nearest_cell(current_state[0], current_state[1])

    update_dashboard(
        x=current_state[0],
        y=current_state[1],
        heading_idx=current_state[2],
        terrain_type=terrain[cur_r, cur_c],
        speed=0.0,
        decision=f"Evaluate options -> {chosen_option.upper()}",
        reason=reason
    )

    push_log(f"Decision safe={score_safe:.1f}")
    push_log(f"Decision fast={score_fast:.1f}")

    map_info.set_text(
        f"KARAR NOKTASI\n"
        f"Safe  -> Cost {safe_cost:.1f}, Risk {safe_risk:.1f}\n"
        f"Fast  -> Cost {fast_cost:.1f}, Risk {fast_risk:.1f}\n"
        f"Rover selected: {chosen_option.upper()}\n"
        f"Devam icin butona bas"
    )

    sim_state = "decision_wait"
    fig.canvas.draw_idle()
    return True

def clear_decision_visuals():
    opt1_line.set_data([], [])
    opt2_line.set_data([], [])
    refresh_visibility_overlay()
    fig.canvas.draw_idle()

# =========================================================
# ANİMASYON
# =========================================================
def terrain_behavior(terrain_type, row=None, col=None):
    if terrain_type == ICE:
        ice_level = 0.0 if row is None else float(ice_strength_map[row, col])
        speed = max(0.05, 0.16 - 0.08 * ice_level)
        energy_cost = 0.35 + 0.35 * ice_level
        return speed, energy_cost, "Speed Reduced", "Ice severity adjusted", "Buzlanma seviyesi yuksek"

    if terrain_type == SOFT:
        sink_level = 0.0 if row is None else float(soft_sink_map[row, col])
        speed = max(0.05, 0.15 - 0.09 * sink_level)
        energy_cost = 0.40 + 0.40 * sink_level
        return speed, energy_cost, "Cautious Traverse", "Sink severity adjusted", "Batma riski yuksek"

    if terrain_type == STEEP:
        if row is not None and col is not None:
            slope_norm = float(slope_map[row, col]) / (float(np.max(slope_map)) + 1e-9)
        else:
            slope_norm = 0.5
        speed = max(0.05, 0.15 - 0.07 * slope_norm)
        energy_cost = 0.42 + 0.30 * slope_norm
        return speed, energy_cost, "Slope Control", "Slope-dependent slowdown", "Egim yuksek"

    return 0.22, 0.22, "Cruise", "Terrain stable", None

def stop_timer():
    global timer
    if timer is not None:
        timer.stop()
        timer = None

def schedule_next(ms=None):
    global timer
    stop_timer()
    interval = FRAME_INTERVAL_MS if ms is None else int(ms)
    timer = fig.canvas.new_timer(interval=interval)
    timer.add_callback(step_simulation)
    timer.start()

def reset_visual_motion():
    global trail_x, trail_y, motion_substep
    trail_x = []
    trail_y = []
    motion_substep = 0
    trail_line.set_data([], [])
    rover_marker.set_data([], [])
    clear_decision_visuals()

def step_simulation():
    global current_index, current_path, sim_state, energy, selection_stage, motion_substep

    if sim_state != "moving":
        return

    if current_index >= len(current_path):
        if current_path is not path_states:
            current_path = path_states
            current_index = 0 if resume_main_index is None else resume_main_index
            motion_substep = 0
            clear_decision_visuals()
            update_orbital_inset()
            schedule_next()
            return

        sim_state = "idle"
        map_info.set_text("Rover hedefe ulasti. Yeni gorev icin START sec.")
        update_dashboard(decision="Stopped", reason="Goal reached", speed=0.0)
        push_log("Mission complete")
        selection_stage = 0
        set_main_view("planning")
        fig.canvas.draw_idle()
        return

    if (
        current_path is path_states
        and current_index > 0
        and motion_substep == 0
        and current_index % DECISION_INTERVAL == 0
        and current_index < len(path_states) - (DECISION_LOOKAHEAD + 2)
    ):
        stop_timer()
        if show_decision_options():
            return
        schedule_next()
        return

    target_state = current_path[current_index]
    if current_index == 0:
        prev_state = target_state
    else:
        prev_state = current_path[current_index - 1]

    tx, ty, th = target_state
    px, py, _ = prev_state

    if current_index == 0:
        ratio = 1.0
    else:
        ratio = (motion_substep + 1) / MOTION_SUBSTEPS

    x = px + (tx - px) * ratio
    y = py + (ty - py) * ratio

    r, c = nearest_cell(tx, ty)
    reveal_info = reveal_local_area(x, y, radius=SENSOR_RADIUS)
    terrain_type = terrain_label_for_display(r, c)
    speed, energy_cost, decision, reason, warning_msg = terrain_behavior(terrain_type, r, c)

    needs_replan, replan_reason = upcoming_path_hazard(current_path, current_index, lookahead=REPLAN_LOOKAHEAD)
    if GOAL is not None and (reveal_info["new_blocked"] or needs_replan) and should_attempt_replan(current_path, current_index, x, y):
        trigger_reason = replan_reason if replan_reason else "New obstacle discovered"
        stop_timer()
        replan_status = replan_from_pose(x, y, th, trigger_reason=trigger_reason)

        if replan_status in ("replanned", "retreat"):
            refresh_visibility_overlay()
            update_orbital_inset()
            fig.canvas.draw_idle()
            schedule_next()
            return

        # fail/cooldown durumunda timer'i kapatıp geri dönmek yerine harekete devam
        sim_state = "moving"

    rover_marker.set_data([x], [y])
    rover_marker.set_marker((3, 0, heading_to_deg(th)))

    trail_x.append(x)
    trail_y.append(y)
    trail_line.set_data(trail_x, trail_y)
    register_safe_pose(x, y, th)
    update_orbital_inset()

    update_dashboard(
        x=x,
        y=y,
        heading_idx=th,
        terrain_type=terrain_type,
        speed=speed,
        decision=decision,
        reason=reason
    )
    refresh_visibility_overlay()

    is_last_substep = (current_index == 0) or (motion_substep >= MOTION_SUBSTEPS - 1)

    if is_last_substep:
        energy = max(0.0, energy - energy_cost)

        if warning_msg and (len(log_lines) == 0 or log_lines[-1] != shorten_log(warning_msg)):
            push_log(warning_msg)

        current_index += 1
        motion_substep = 0
    else:
        motion_substep += 1

    fig.canvas.draw_idle()
    schedule_next()

# =========================================================
# HARİTAYA TIKLAMA
# =========================================================
def on_map_click(event):
    global START, GOAL, selection_stage, start_scatter, goal_scatter
    global sim_state, current_path, current_index, energy, mission_report, last_replan_time
    global last_replan_fail_time, last_replan_fail_cell, safe_history

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    if sim_state == "moving":
        return

    x = clamp(float(event.xdata), 0, COLS - 1)
    y = clamp(float(event.ydata), 0, ROWS - 1)

    if selection_stage == 0:
        validation = validate_point(x, y, point_name="START")
        map_info.set_text(validation["message"])

        if not validation["valid"]:
            update_dashboard(decision="START Rejected", reason="Invalid placement", speed=0.0)
            push_log("START rejected")
            fig.canvas.draw_idle()
            return

        stop_timer()
        sim_state = "idle"
        energy = 100.0
        GOAL = None
        mission_report = None
        START = (x, y)
        selection_stage = 1
        last_replan_time = -1e9
        last_replan_fail_time = -1e9
        last_replan_fail_cell = None
        safe_history = []
        initialize_known_maps()
        reveal_local_area(START[0], START[1], radius=SENSOR_RADIUS)
        register_safe_pose(START[0], START[1], 0)
        update_lidar_panel(START[0], START[1], 0)

        current_path = []
        current_index = 0
        path_line.set_data([], [])
        trail_line.set_data([], [])
        rover_marker.set_data([], [])

        if start_scatter is not None:
            start_scatter.remove()
            start_scatter = None
        if goal_scatter is not None:
            goal_scatter.remove()
            goal_scatter = None

        start_scatter = ax.scatter(
            START[0], START[1],
            c="deepskyblue", s=140,
            edgecolors="black", zorder=12
        )

        if validation["severity"] == "yellow":
            push_log("START accepted warn")
        else:
            push_log("START accepted")

        set_main_view("planning")
        refresh_visibility_overlay()
        update_orbital_inset()
        update_dashboard(decision="Awaiting GOAL", reason="Mission control selecting target", speed=0.0)
        fig.canvas.draw_idle()

    elif selection_stage == 1:
        validation = validate_point(x, y, point_name="GOAL")
        map_info.set_text(validation["message"])

        if not validation["valid"]:
            update_dashboard(decision="GOAL Rejected", reason="Invalid placement", speed=0.0)
            push_log("GOAL rejected")
            fig.canvas.draw_idle()
            return

        GOAL = (x, y)
        selection_stage = 2

        if goal_scatter is not None:
            goal_scatter.remove()

        goal_scatter = ax.scatter(
            GOAL[0], GOAL[1],
            c="red", s=160, marker="X",
            edgecolors="black", zorder=12
        )

        fig.canvas.draw_idle()

        ok = recompute_main_path()
        if ok:
            current_path = path_states
            current_index = 0
            path_line.set_data([], [])
            reset_visual_motion()

            if path_states:
                rover_marker.set_data([path_states[0][0]], [path_states[0][1]])
                rover_marker.set_marker((3, 0, heading_to_deg(path_states[0][2])))

            msg = validation["message"] + "\n\n" + mission_report_text(mission_report)
            msg += f"\n\nPlanner: {last_planner_mode}"

            if mission_report["status"] == "red":
                msg += "\n\nRecharge required before mission."
            elif mission_report["status"] == "yellow":
                msg += "\n\nLow reserve: recharge recommended."
            else:
                msg += "\n\nMission is ready."

            map_info.set_text(msg)
            update_dashboard(decision="Path Ready", reason="Mission analyzed", speed=0.0)

            if validation["severity"] == "yellow":
                push_log("GOAL accepted warn")
            else:
                push_log("GOAL accepted")

            push_log(f"Energy {mission_report['energy_required']:.1f}%")
            push_log(last_planner_mode)
            update_orbital_inset()
        else:
            map_info.set_text("Rota bulunamadi. Yeni START sec.")
            update_dashboard(decision="Path Failed", reason="No traversable route", speed=0.0)
            push_log("Route failed")
            selection_stage = 0
            START = None
            GOAL = None
            mission_report = None

            if start_scatter is not None:
                start_scatter.remove()
                start_scatter = None
            if goal_scatter is not None:
                goal_scatter.remove()
                goal_scatter = None
            set_main_view("planning")
            update_orbital_inset()

        fig.canvas.draw_idle()

    else:
        selection_stage = 0
        set_main_view("planning")
        map_info.set_text("MISSION PLANNING\nYeni gorev: kaba haritada START sec.")
        update_orbital_inset()
        fig.canvas.draw_idle()
        on_map_click(event)

fig.canvas.mpl_connect("button_press_event", on_map_click)

# =========================================================
# BUTON
# =========================================================
def on_button(event):
    global sim_state, current_path, current_index, motion_substep

    if START is None or GOAL is None or not path_states:
        map_info.set_text("Once haritadan gecerli START ve GOAL sec.")
        update_dashboard(decision="Waiting", reason="Select valid mission points", speed=0.0)
        fig.canvas.draw_idle()
        return

    if mission_report is not None and mission_report["status"] == "red" and sim_state == "idle":
        map_info.set_text(
            mission_report_text(mission_report) +
            "\n\nMission blocked: battery insufficient.\nRecharge required before start."
        )
        update_dashboard(decision="Mission Blocked", reason="Battery insufficient", speed=0.0)
        fig.canvas.draw_idle()
        return

    if sim_state == "decision_wait":
        chosen_seg = decision_option_safe if chosen_option == "safe" else decision_option_fast
        current_path = chosen_seg
        current_index = 0
        motion_substep = 0
        clear_decision_visuals()
        map_info.set_text(f"Rover {chosen_option.upper()} rotayi secti. Devam ediyor...")
        push_log(f"Selected {chosen_option}")
        sim_state = "moving"
        schedule_next()
        return

    if sim_state == "moving":
        return

    if sim_state == "idle" and len(path_x) > 0 and len(path_line.get_xdata()) == 0:
        set_main_view("rover")
        path_line.set_data(path_x, path_y)
        map_info.set_text("ROVER VIEW\nAna ekran artik rover algisinda.\nKosedeki kutu kaba orbital haritayi gosterir.\nTekrar bas: rover hareket etsin.")
        update_dashboard(decision="Route Shown", reason="Awaiting start", speed=0.0)
        fig.canvas.draw_idle()
        return

    if sim_state == "idle":
        current_path = path_states
        current_index = 0
        motion_substep = 0
        reset_visual_motion()
        path_line.set_data(path_x, path_y)
        sim_state = "moving"
        map_info.set_text("Rover hareket ediyor...")
        push_log("Simulation started")
        schedule_next()

button_ax = fig.add_axes([0.72, 0.015, 0.24, 0.040])
button = Button(button_ax, "Plan / Start / Continue")
button.on_clicked(on_button)

plt.show()