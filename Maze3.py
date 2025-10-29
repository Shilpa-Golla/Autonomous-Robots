from controller import Supervisor
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import heapq


TIME_STEP = 32
GRID_RES = 0.05
LO_OCC = np.log(0.7/0.3)
LO_FREE = np.log(0.3/0.7)
LO_MIN = -8.0
LO_MAX = 8.0

robot = Supervisor()
root = robot.getRoot()

children_field  = root.getField('children')

top_level_nodes = [children_field.getMFNode(i)

                   for i in range(children_field.getCount())]

def find_node_by_name(node, target_name):
    name_field = node.getField("name")
    if name_field and name_field.getSFString() == target_name:
        return node

    children = node.getField("children")
    if children:
        for i in range(children.getCount()):
            found = find_node_by_name(children.getMFNode(i), target_name)
            if found:
                return found
    return None

def find_node_by_type(node, target_type):
    if node.getTypeName() == target_type:
        return node
    children = node.getField("children")
    if children:
        for i in range(children.getCount()):
            found = find_node_by_type(children.getMFNode(i), target_type)
            if found:
                return found
    return None

def find_node_by_name_global(target_name):
    for n in top_level_nodes:
        found = find_node_by_name(n, target_name)
        if found:
            return found
    return None

def find_node_by_type_global(target_type):
    for n in top_level_nodes:
        found = find_node_by_type(n, target_type)
        if found:
            return found
    return None

blue_node   = find_node_by_name_global('BlueCylinder')
yellow_node = find_node_by_name_global('YellowCylinder')
floor_node  = find_node_by_type_global('Floor')
robot_node  = find_node_by_type_global('Rosbot')

motors = [robot.getDevice(name) for name in ['fl_wheel_joint','fr_wheel_joint','rl_wheel_joint','rr_wheel_joint']]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0)
lidar = robot.getDevice('laser')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()
camera = robot.getDevice('camera rgb')
camera.enable(TIME_STEP)


blue_pos   = blue_node.getField('translation').getSFVec3f()
yellow_pos = yellow_node.getField('translation').getSFVec3f()

floor_size = floor_node.getField('size').getSFVec3f()
world_width, world_length = floor_size[0], floor_size[1]
floor_pos = floor_node.getField('translation').getSFVec3f()
GRID_W = int(world_width  / GRID_RES)
GRID_H = int(world_length / GRID_RES)
ORIGIN = [floor_pos[0] - world_width/2,
          floor_pos[1] - world_length/2]

occupancy_lo = np.zeros((GRID_H, GRID_W), dtype=np.float32)  # log-odds

def lo_to_prob(lo):
    return 1.0 - 1.0 / (1.0 + np.exp(lo))

def world_to_grid(x, y):
    gx = int((x - ORIGIN[0]) / GRID_RES)
    gy = int((y - ORIGIN[1]) / GRID_RES)
    return gx, gy

def grid_to_world(gx, gy):
    x = ORIGIN[0] + gx * GRID_RES
    y = ORIGIN[1] + gy * GRID_RES
    return x, y

def bresenham(x0, y0, x1, y1):
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    cells.append((x1, y1))
    return cells
# Reference https://chatgpt.com/share/6883fb28-7df4-8013-b113-fb772d0612b1

def mark_lidar_ray(robot_x, robot_y, hit_x, hit_y, hit_is_wall, beam_valid):
    gx0, gy0 = world_to_grid(robot_x, robot_y)
    gx1, gy1 = world_to_grid(hit_x, hit_y)
    cells = bresenham(gx0, gy0, gx1, gy1)
    if beam_valid:
        for cx, cy in cells[:-1]:
            if 0 <= cx < GRID_W and 0 <= cy < GRID_H:
                occupancy_lo[cy, cx] = np.clip(occupancy_lo[cy, cx] + LO_FREE, LO_MIN, LO_MAX)
        if hit_is_wall and 0 <= cells[-1][0] < GRID_W and 0 <= cells[-1][1] < GRID_H:
            occupancy_lo[cells[-1][1], cells[-1][0]] = np.clip(occupancy_lo[cells[-1][1], cells[-1][0]] + LO_OCC, LO_MIN, LO_MAX)
    else:
        # Don't mark any cell as free if beam was invalid (unless you know for sure)
        pass

def set_wheel_vel(left, right):
    motors[0].setVelocity(left)
    motors[1].setVelocity(right)
    motors[2].setVelocity(left)
    motors[3].setVelocity(right)

def is_red_detected(img, w, h, red_thresh=80):
    start_x = w // 3
    end_x = 2 * w // 3
    red_pixels = 0
    total_pixels = (end_x - start_x) * h
    for x in range(start_x, end_x):
        for y in range(h):
            idx = 3 * (y * w + x)
            r = img[idx + 0]
            g = img[idx + 1]
            b = img[idx + 2]
            if r > 120 and g < red_thresh and b < red_thresh:
                red_pixels += 1
    return red_pixels > (0.1 * total_pixels)
    
def mark_wall_across_corridor(gx, gy, heading, max_length=30, thickness=3):
    perp_angle = heading + np.pi/2
    half_thick = thickness // 2
    # Mark the center cell and a band perpendicular to heading
    for offset in range(-half_thick, half_thick+1):
        ox = int(round(offset * np.cos(heading)))
        oy = int(round(offset * np.sin(heading)))
        cx = gx + ox
        cy = gy + oy
        for sign in [-1, 1]:  # Both sides from center
            for step in range(max_length):
                dx = int(round(sign * step * np.cos(perp_angle)))
                dy = int(round(sign * step * np.sin(perp_angle)))
                sx = cx + dx
                sy = cy + dy
                if not (0 <= sx < occupancy_lo.shape[1] and 0 <= sy < occupancy_lo.shape[0]):
                    break
                if lo_to_prob(occupancy_lo)[sy, sx] > 0.65:  # hit a wall
                    break
                occupancy_lo[sy, sx] = LO_MAX
        # Always mark center point
        if 0 <= cx < occupancy_lo.shape[1] and 0 <= cy < occupancy_lo.shape[0]:
            occupancy_lo[cy, cx] = LO_MAX

# Reference: Chatgpt code

FORWARD_SPEED = 8.0
TURN_SPEED = 4.0

# plt.ion()
# fig, ax = plt.subplots(figsize=(6,6))
# extent = [0, GRID_W, 0, GRID_H]
# im = ax.imshow(lo_to_prob(occupancy_lo), cmap='gray_r', vmin=0, vmax=1, origin='lower', extent=extent)
# robot_dot, = ax.plot([], [], 'ro', markersize=8)
# frontier_dot, = ax.plot([], [], 'bo', markersize=6)
# ax.set_title("Occupancy Grid (log-odds)")
# plt.show(block=False)
path=None

def find_frontier_cells(occupancy_lo, occ_threshold=0.6):
    unknown = np.abs(occupancy_lo) < 1e-2
    occ_prob = lo_to_prob(occupancy_lo)
    free = occ_prob < 0.35
    structure = np.ones((3, 3), dtype=bool)
    free_dilated = scipy.ndimage.binary_dilation(free, structure=structure)
    frontier_mask = unknown & free_dilated
    return frontier_mask

def update_clearance_map(occupancy_bin):
    clearance = distance_transform_edt(occupancy_bin == 0)
    return clearance

def astar_path_clearance(start, goal, occupancy_bin, clearance_map, robot_radius_cells):
    h = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance
    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(-1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if not (0 <= x < occupancy_bin.shape[1] and 0 <= y < occupancy_bin.shape[0]):
                continue
            if occupancy_bin[y, x] == 1 or clearance_map[y, x] < robot_radius_cells:
                continue
            tentative_g = g_score[current] + (1.0 if dx == 0 or dy == 0 else 1.414)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current
    return None


def bfs_path_clearance(start, goal, occupancy_bin, clearance_map, robot_radius_cells):
    visited = set()
    queue = deque()
    prev = dict()
    queue.append(start)
    visited.add(start)
    while queue:
        current = queue.popleft()
        if current == goal:
            path = [current]
            while current != start:
                current = prev[current]
                path.append(current)
            path.reverse()
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(-1,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if (0 <= nx < occupancy_bin.shape[1] and 0 <= ny < occupancy_bin.shape[0] and
                occupancy_bin[ny, nx] == 0.0 and (nx, ny) not in visited and
                clearance_map[ny, nx] >= robot_radius_cells):
                queue.append((nx, ny))
                visited.add((nx, ny))
                prev[(nx, ny)] = current
    return None

def go_towards(robot_x, robot_y, robot_heading, target_x, target_y):
    dx = target_x - robot_x
    dy = target_y - robot_y
    target_angle = math.atan2(dy, dx)
    angle_diff = (target_angle - robot_heading + math.pi) % (2*math.pi) - math.pi
    dist = np.hypot(dx, dy)
    if dist < 0.07: 
        set_wheel_vel(0, 0)
        return
    # Find the angle difference from a cell to the target cell and we tune the degree and make the robot move accordingly
    max_forward = 8.0 
    max_turn = 4.0
    if abs(angle_diff) < np.deg2rad(25):
        forward = max_forward
    elif abs(angle_diff) < np.deg2rad(60):
        forward = max_forward * (1 - abs(angle_diff) / np.deg2rad(60))
    else:
        forward = 0.0 

    turn = 2.5 * angle_diff  # Reduce this value for less "wiggle"
    turn = np.clip(turn, -max_turn, max_turn)

    left = np.clip(forward - turn, -max_forward, max_forward)
    right = np.clip(forward + turn, -max_forward, max_forward)
    set_wheel_vel(left, right)
    
def is_path_valid(path, occupancy_for_path, clearance_map, robot_radius_cells):
    for gx, gy in path:
        if (gx < 0 or gx >= occupancy_for_path.shape[1] or
            gy < 0 or gy >= occupancy_for_path.shape[0]):
            return False  # Out of bounds
        if occupancy_for_path[gy, gx] == 1:  # Occupied
            return False
        if clearance_map[gy, gx] < robot_radius_cells:
            return False  # Too narrow to pass
    return True

    
def select_best_reachable_frontier(robot_grid, frontier_mask, occupancy_lo, occupancy_for_path, clearance_map, robot_radius_cells):
    frontiers = np.argwhere(frontier_mask)
    reachable = []
    gx, gy = robot_grid
    for fy, fx in frontiers:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(1,1),(-1,1)]:
            nx, ny = fx + dx, fy + dy
            if (0 <= nx < occupancy_lo.shape[1] and 0 <= ny < occupancy_lo.shape[0]
                and lo_to_prob(occupancy_lo)[ny, nx] < 0.35
                and occupancy_for_path[ny, nx] == 0.0
                and clearance_map[ny, nx] >= robot_radius_cells):
                path = bfs_path_clearance((gx, gy), (nx, ny), occupancy_for_path, clearance_map, robot_radius_cells)
                if path is not None:
                    distance = len(path)
                    reachable.append(((nx, ny), (fx, fy), distance))
                break 

    if not reachable:
        return None, None
    reachable.sort(key=lambda x: x[2])
    return reachable[0][0], reachable[0][1]
    
def compute_reachable_mask(start, occupancy_bin, clearance_map, robot_radius_cells):
    reachable = np.zeros_like(occupancy_bin, dtype=bool)
    queue = deque()
    x0, y0 = start
    if not (0 <= x0 < occupancy_bin.shape[1] and 0 <= y0 < occupancy_bin.shape[0]):
        return reachable
    queue.append((x0, y0))
    while queue:
        x, y = queue.popleft()
        if not (0 <= x < occupancy_bin.shape[1] and 0 <= y < occupancy_bin.shape[0]):
            continue
        if reachable[y, x]:
            continue
        if occupancy_bin[y, x] == 1 or clearance_map[y, x] < robot_radius_cells:
            continue
        reachable[y, x] = True
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            queue.append((x+dx, y+dy))
    return reachable
    
def extract_frontier_regions(frontier_mask, min_size=8):
    structure = np.ones((3,3), dtype=bool)
    labeled, num_features = scipy.ndimage.label(frontier_mask, structure=structure)
    regions = []
    for i in range(1, num_features+1):
        cells = np.argwhere(labeled == i)  # (y, x)
        if len(cells) >= min_size:
            regions.append(cells)
    return regions

def region_centroid(region):
    y, x = np.mean(region, axis=0)
    return int(round(x)), int(round(y))


target_frontier = None
best_target = None
path = None
phase='explore'

def find_nearest_reachable_free(goal_gx, goal_gy, occupancy_bin, clearance_map, robot_radius_cells, max_search_radius=8):
    best = None
    best_dist = float('inf')
    for r in range(1, max_search_radius+1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = goal_gx + dx, goal_gy + dy
                if (0 <= nx < occupancy_bin.shape[1] and 0 <= ny < occupancy_bin.shape[0]):
                    if occupancy_bin[ny, nx] == 0 and clearance_map[ny, nx] >= robot_radius_cells:
                        d = abs(dx) + abs(dy)
                        if d < best_dist:
                            best = (nx, ny)
                            best_dist = d
        if best is not None:
            return best
    return (goal_gx, goal_gy)



while robot.step(TIME_STEP) != -1:
    # --- Pose and heading ---
    rpos = robot_node.getField('translation').getSFVec3f()
    pose_x, pose_y = rpos[0], rpos[1]
    rot_field = robot_node.getField('rotation').getSFRotation()
    heading = rot_field[3] * rot_field[2]

    # --- Lidar update ---
    lidar_vals = lidar.getRangeImage()
    n = len(lidar_vals)
    fov = lidar.getFov()
    angle_res = fov / (n - 1)
    max_range = lidar.getMaxRange()
    for i in range(n):
        dist = lidar_vals[i]
        if math.isinf(dist) or math.isnan(dist):
            continue
        angle = fov/2 - i * angle_res
        global_angle = heading + angle
        hit_x = pose_x + dist * math.cos(global_angle)
        hit_y = pose_y + dist * math.sin(global_angle)
        hit_is_wall = (dist < max_range * 0.99)
        mark_lidar_ray(pose_x, pose_y, hit_x, hit_y, hit_is_wall, beam_valid=True)

    # --- Robot grid location ---
    gx, gy = world_to_grid(pose_x, pose_y)
    # robot_dot.set_data([gx], [gy])
    if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
        occupancy_lo[gy, gx] = np.clip(occupancy_lo[gy, gx] + LO_FREE, LO_MIN, LO_MAX)

    # --- Camera: red wall detection ---
    camera_img = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    red_detected = is_red_detected(camera_img, w, h)

    # --- Obstacle detection ---
    center = n // 2
    window = n // 16
    front_vals = lidar_vals[center - window: center + window]
    blocked = any(d < 0.3 for d in front_vals)
    move_blocked = red_detected

    # --- Occupancy or clearance maps ---
    robot_radius_m = 0.21
    robot_radius_cells = int(np.ceil(robot_radius_m / GRID_RES))
    occ_prob = lo_to_prob(occupancy_lo)
    occupancy_for_path = (occ_prob > 0.65).astype(np.uint8)  # 1 = occupied, 0 = free
    clearance_map = update_clearance_map(occupancy_for_path)
    robot_grid = (gx, gy)

    # --- Visualization ---
    prob_grid = lo_to_prob(occupancy_lo)
    unknown_mask = np.abs(occupancy_lo) < 1e-2
    occupied_mask = prob_grid > 0.65
    free_mask = prob_grid < 0.35
    vis_grid = np.zeros_like(occupancy_lo, dtype=np.float32)
    vis_grid[unknown_mask] = 0.5   # Gray for unknown
    vis_grid[occupied_mask] = 0.0  # Black for occupied
    vis_grid[free_mask] = 1.0      # White for free
    
    percent_free = round(100 * np.sum(free_mask) / occupancy_lo.size, 2)
    print(f"Percent of mapped area that is free: {percent_free:.2f}%")
    
    if(percent_free > 8.0 and phase=='explore'):
        phase = 'toBlue'
        best_path = None
            
    # im.set_data(vis_grid)
    # im.set_cmap('gray')
    # im.set_clim(0, 1)

    if not move_blocked:
        if phase =='toBlue': 
            print("Navigate to blue pillar.")
            target_gx, target_gy = world_to_grid(blue_pos[0], blue_pos[1])
            best_goal = find_nearest_reachable_free(target_gx, target_gy, occupancy_for_path, clearance_map, robot_radius_cells)            
            occupancy_for_path = ((occ_prob > 0.65) & (np.abs(occupancy_lo) > 1e-2)).astype(np.uint8)
            clearance_map = update_clearance_map(occupancy_for_path)
            
            best_path = astar_path_clearance((gx, gy), best_goal, occupancy_for_path, clearance_map, robot_radius_cells)
            if best_path is not None and len(best_path) > 4:
                next_gx, next_gy = best_path[1]
                go_towards(gx, gy, heading, next_gx, next_gy)
                if (gx, gy) == (next_gx, next_gy):
                    best_path.pop(0)
            elif best_path is not None and len(best_path) <=4: 
                 phase = 'toYellow'
            else:
                print("No valid path to blue pillar.")
                if 'stuck_counter' not in locals():
                    stuck_counter = 0
                
                if blocked:
                    stuck_counter += 1
                    # Analyze lidar for open space direction
                    n = len(lidar_vals)
                    sectors = 7 
                    sector_size = n // sectors
                    avg_ranges = []
                    for i in range(sectors):
                        sector_ranges = lidar_vals[i*sector_size : (i+1)*sector_size]
                        sector_ranges = [r for r in sector_ranges if not (math.isinf(r) or math.isnan(r))]
                        avg = np.mean(sector_ranges) if sector_ranges else 0.0
                        avg_ranges.append(avg)
                    best_sector = np.argmax(avg_ranges)
                
                    if stuck_counter > 15:
                        set_wheel_vel(-FORWARD_SPEED, -FORWARD_SPEED) 
                        robot.step(TIME_STEP * 10)
                        if np.random.rand() > 0.5:
                            set_wheel_vel(TURN_SPEED, -TURN_SPEED)
                        else:
                            set_wheel_vel(-TURN_SPEED, TURN_SPEED)
                        robot.step(TIME_STEP * 12)
                        stuck_counter = 0
                        continue
                
                    if best_sector == sectors // 2:
                        set_wheel_vel(FORWARD_SPEED * 0.5, FORWARD_SPEED * 0.5)
                    elif best_sector < sectors // 2:
                        set_wheel_vel(-TURN_SPEED, TURN_SPEED)
                    else:
                        set_wheel_vel(TURN_SPEED, -TURN_SPEED)
                    continue
                else:
                    stuck_counter = 0
                    set_wheel_vel(FORWARD_SPEED * 0.5, FORWARD_SPEED * 0.5)
                    continue

        elif phase =='toYellow': 
            print("Navigate to yellow pillar")
            target_gx, target_gy = world_to_grid(yellow_pos[0], yellow_pos[1])
            best_goal = find_nearest_reachable_free(target_gx, target_gy, occupancy_for_path, clearance_map, robot_radius_cells)            
            occupancy_for_path = ((occ_prob > 0.65) & (np.abs(occupancy_lo) > 1e-2)).astype(np.uint8)           
            clearance_map = update_clearance_map(occupancy_for_path)
            
            best_path = astar_path_clearance((gx, gy), best_goal, occupancy_for_path, clearance_map, robot_radius_cells)
            
            if best_path is not None and len(best_path) > 4:
                next_gx, next_gy = best_path[1]
                go_towards(gx, gy, heading, next_gx, next_gy)
            elif best_path is not None and len(best_path) <=4: 
                 set_wheel_vel(0, 0)
                 print("Successfully Completed")
            else:
                print("No valid path to yellow pillar.")
                if blocked:
                    n = len(lidar_vals)
                    sectors = 7 
                    sector_size = n // sectors
                    avg_ranges = []
                
                    for i in range(sectors):
                        sector_ranges = lidar_vals[i*sector_size : (i+1)*sector_size]
                        sector_ranges = [r for r in sector_ranges if not (math.isinf(r) or math.isnan(r))]
                        avg = np.mean(sector_ranges) if sector_ranges else 0.0
                        avg_ranges.append(avg)
                
                    best_sector = np.argmax(avg_ranges)
                    sector_angle = (best_sector - (sectors // 2)) * (lidar.getFov() / sectors)
                
                    # If the best sector is in front (center), go forward
                    if best_sector == sectors // 2:
                        set_wheel_vel(FORWARD_SPEED * 0.5, FORWARD_SPEED * 0.5)
                    elif best_sector < sectors // 2:
                        # Best sector is to the left, turn left
                        set_wheel_vel(-FORWARD_SPEED * 0.2, FORWARD_SPEED * 0.2)
                    else:
                        # Best sector is to the right, turn right
                        set_wheel_vel(FORWARD_SPEED * 0.2, -FORWARD_SPEED * 0.2)
                    continue
                else:
                    # Not blocked: go forward (or your normal code)
                    set_wheel_vel(FORWARD_SPEED * 0.5, FORWARD_SPEED * 0.5)
                    continue
        elif phase == 'explore':
            reachable_mask = compute_reachable_mask(robot_grid, occupancy_for_path, clearance_map, robot_radius_cells)
            frontier_mask = find_frontier_cells(occupancy_lo)
    
            regions = extract_frontier_regions(frontier_mask, min_size=8)
    
            best_path = None
            best_distance = float('inf')
            best_region = None
            best_region_centroid = None
            best_region_target = None
    
            for region in regions:
                cx, cy = region_centroid(region)
                region_target = None
                region_path = None
                for y, x in region:
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < occupancy_lo.shape[1] and 0 <= ny < occupancy_lo.shape[0]:
                            if reachable_mask[ny, nx]:
                                # Try to path to this cell
                                path = bfs_path_clearance((gx, gy), (nx, ny), occupancy_for_path, clearance_map, robot_radius_cells)
                                if path is not None and len(path) < best_distance:
                                    best_distance = len(path)
                                    best_path = path
                                    best_region = region
                                    best_region_centroid = (cx, cy)
                                    best_region_target = (nx, ny)
                                break
                    if best_region_target:
                        break 
    
            if best_region is None:
                print("[INFO] No reachable frontier regions at current safety settings. Trying recovery...")
                recovery_robot_radius_cells = max(1, int(robot_radius_cells * 0.7))  # Reduce by 30%, but at least 1 cell
                print(f"[RECOVERY] Reducing robot radius from {robot_radius_cells} to {recovery_robot_radius_cells} cells")
                recovery_reachable_mask = compute_reachable_mask(robot_grid, occupancy_for_path, clearance_map, recovery_robot_radius_cells)
            
                best_path = None
                best_distance = float('inf')
                best_region = None
                best_region_centroid = None
                best_region_target = None
            
                for region in regions:
                    cx, cy = region_centroid(region)
                    for y, x in region:
                        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < occupancy_lo.shape[1] and 0 <= ny < occupancy_lo.shape[0]:
                                if recovery_reachable_mask[ny, nx]:
                                    path = bfs_path_clearance((gx, gy), (nx, ny), occupancy_for_path, clearance_map, recovery_robot_radius_cells)
                                    if path is not None and len(path) < best_distance:
                                        best_distance = len(path)
                                        best_path = path
                                        best_region = region
                                        best_region_centroid = (cx, cy)
                                        best_region_target = (nx, ny)
                                    break
                        if best_region_target:
                            break
            
                if best_region is not None and best_path is not None and len(best_path) > 1:
                    print("[RECOVERY] Found new reachable region with relaxed margin.")
                    fx, fy = best_region_centroid
                    # frontier_dot.set_data([fx], [fy])
                    next_gx, next_gy = best_path[1]
                    go_towards(gx, gy, heading, next_gx, next_gy)
                    continue
            
                # --- If STILL nothing, really stop ---
                num_free = np.sum((lo_to_prob(occupancy_lo) < 0.35) & (np.abs(occupancy_lo) > 1e-2))
                if num_free < 190:
                    set_wheel_vel(FORWARD_SPEED * 0.2, FORWARD_SPEED * 0.2)
                    print("Waiting for map to initialize...")
                    continue
                set_wheel_vel(-TURN_SPEED, -TURN_SPEED)
                print("Exploration complete or no reachable frontier regions, even in recovery mode!")
                phase='toBlue'
                continue
    
            fx, fy = best_region_centroid
            # frontier_dot.set_data([fx], [fy])
    
            # 5. Move toward the reachable target cell in the region
            if best_path is not None and len(best_path) > 1:
                next_gx, next_gy = best_path[1]
                go_towards(gx, gy, heading, next_gx, next_gy)
            else:
                set_wheel_vel(TURN_SPEED, -TURN_SPEED)
                print("No valid path to region frontier neighbor!")

    else:
        if red_detected:
            set_wheel_vel(TURN_SPEED, -TURN_SPEED)

    # if hasattr(robot, 'last_path_line'):
        # robot.last_path_line.remove()
        # delattr(robot, 'last_path_line')
    # if best_path is not None and len(best_path) > 1:
        # path_x = [p[0] for p in best_path]
        # path_y = [p[1] for p in best_path]
        # path_line, = ax.plot(path_x, path_y, 'c-', linewidth=2, alpha=0.7)
        # robot.last_path_line = path_line
    # fig.canvas.flush_events()
    # plt.pause(0.001)

    if int(robot.getTime()) % 10 == 0:
        np.save('occupancy_grid.npy', lo_to_prob(occupancy_lo))
