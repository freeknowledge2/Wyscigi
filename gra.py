# Race Track Simulation with Drivers
import pygame
import random
import math
import time
import json
import os

# --- CONFIG ---
# default (overridden on init)
WIDTH, HEIGHT = 900, 600  # will be overridden to fullscreen after init
TRACK_MARGIN = 80
OBSTACLE_COUNT = 28
DRIVER_COUNT = 10
MAX_TIME = 60
DRIVER_RADIUS = 12
FPS = 60
MAZE_MODE = True
# how long a tire mark stays on track (seconds)
MARK_LIFETIME = 3.0
# track width (distance between outer and inner edges) - larger for realistic track
TRACK_WIDTH = 300
# placeholder; will be set after fullscreen init
FINISH_ZONE = pygame.Rect(0, 0, 0, 0)
DRIVER_COLORS = [
	(94, 60, 32),   # brown
	(34, 85, 28),   # forest green
	(52, 101, 164), # slate blue
	(170, 120, 80), # clay
	(120, 120, 120),# stone gray
	(200, 170, 110),# sand
	(150, 30, 40),  # maroon
	(72, 120, 60),  # moss
	(90, 140, 170), # teal
	(110, 80, 60)   # walnut
]
DRIVER_NAMES = [f"Driver {i+1}" for i in range(DRIVER_COUNT)]

pygame.init()
# run in windowed mode for easier testing; set to False to use fullscreen
WINDOWED = True
# default window size when WINDOWED=True (larger for a bigger track)
WINDOW_SIZE = (1600, 900)
if WINDOWED:
	WIDTH, HEIGHT = WINDOW_SIZE
	screen = pygame.display.set_mode((WIDTH, HEIGHT))
else:
	# Switch to fullscreen and update WIDTH/HEIGHT to match the display
	info = pygame.display.Info()
	WIDTH, HEIGHT = info.current_w, info.current_h
	# use hardware surface + double buffering for more reliable fullscreen rendering
	screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Race Track Simulation")
# persistent surface for marks (tire tracks) with alpha
track_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
# ensure same pixel format and alpha handling as display
track_surface = track_surface.convert_alpha()
# active tire marks: list of dicts {id, pos, t, color}
marks_list = []
font = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()

# learning persistence file
LEARNING_FILE = os.path.join(os.path.dirname(__file__), 'learning.json')

def load_learning():
	try:
		if os.path.exists(LEARNING_FILE):
			with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
				return json.load(f)
	except Exception:
		pass
	return {}

def save_learning(data):
	try:
		with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
			json.dump(data, f)
	except Exception:
		pass

# --- OBSTACLES ---
def generate_obstacles(finish_zone, inner_rect, outer_rect):
	obstacles = []
	# Create a few chicanes (pairs of obstacles narrowing the lane)
	num_chicanes = max(2, OBSTACLE_COUNT // 8)
	for i in range(num_chicanes):
		# place along the long edges alternating
		frac = (i+1) / (num_chicanes+1)
		x_center = int(outer_rect.left + frac * outer_rect.width)
		# place chicane across the track vertically
		# increase gap between paired chicane obstacles so lane is less narrow
		gap = 120
		w = 30
		h = 80
		left_rect = pygame.Rect(x_center - gap//2 - w, outer_rect.centery - h//2, w, h)
		right_rect = pygame.Rect(x_center + gap//2, outer_rect.centery - h//2, w, h)
		if not left_rect.colliderect(finish_zone) and not right_rect.colliderect(finish_zone):
			obstacles.append(left_rect)
			obstacles.append(right_rect)
	# then scatter random obstacles on lane
	count = 0
	while count < OBSTACLE_COUNT:
		attempts = 0
		while True:
			attempts += 1
			# smaller random obstacles to increase free space
			w = random.randint(30, 70)
			h = random.randint(20, 60)
			x = random.randint(outer_rect.left, outer_rect.right - w)
			y = random.randint(outer_rect.top, outer_rect.bottom - h)
			rect = pygame.Rect(x, y, w, h)
			# must be on track area (inside outer, outside inner) and not on finish_zone and not overlapping existing
			if outer_rect.colliderect(rect) and not inner_rect.colliderect(rect) and not rect.colliderect(finish_zone) and not any(rect.colliderect(o) for o in obstacles):
				obstacles.append(rect)
				count += 1
				break
			if attempts > 80:
				break
	return obstacles


def generate_maze_and_waypoints(finish_zone, inner_rect, outer_rect, cols=24, rows=14):
	"""Generate a finer grid-maze with multiple path options and dense waypoints
	for better bot navigation and pathfinding performance.
	"""
	# compute cell size to fit in outer_rect minus margins - use finer grid
	cell_w = outer_rect.width / cols
	cell_h = outer_rect.height / rows
	# clamp a minimum cell size
	cell_size = int(max(32, min(cell_w, cell_h)))
	cols = max(8, int(outer_rect.width // cell_size))
	rows = max(6, int(outer_rect.height // cell_size))
	cell_w = outer_rect.width / cols
	cell_h = outer_rect.height / rows
	# grid of walls: True means wall present
	grid = [[True for _ in range(cols)] for __ in range(rows)]
	# carve a maze using randomized DFS starting from left-middle cell
	start_r = rows // 2
	start_c = 1
	stack = [(start_r, start_c)]
	grid[start_r][start_c] = False
	# neighbor offsets (r,c)
	neis = [(0,2),(0,-2),(2,0),(-2,0)]
	import random as _r
	while stack:
		r, c = stack[-1]
		_candidates = []
		for dr, dc in neis:
			r2, c2 = r+dr, c+dc
			if 0 <= r2 < rows and 0 <= c2 < cols and grid[r2][c2]:
				_candidates.append((r2, c2, dr//2, dc//2))
		if _candidates:
			r2, c2, midr, midc = _r.choice(_candidates)
			# remove wall between
			grid[r+midr][c+midc] = False
			grid[r2][c2] = False
			stack.append((r2, c2))
		else:
			stack.pop()
	
	# Create multiple wider corridors for better navigation
	for _ in range(max(2, cols//8)):
		r = _r.randrange(1, rows-1)
		c = _r.randrange(1, cols-1)
		# widen corridors by removing additional walls
		for dr in [-1, 0, 1]:
			for dc in [-1, 0, 1]:
				nr, nc = r+dr, c+dc
				if 0 <= nr < rows and 0 <= nc < cols:
					grid[nr][nc] = False
	
	# Determine finish cell near finish_zone center
	fx = max(outer_rect.left, min(outer_rect.right-1, finish_zone.centerx))
	fy = max(outer_rect.top, min(outer_rect.bottom-1, finish_zone.centery))
	finish_c = int((fx - outer_rect.left) / cell_w)
	finish_r = int((fy - outer_rect.top) / cell_h)
	finish_c = max(0, min(cols-1, finish_c))
	finish_r = max(0, min(rows-1, finish_r))
	
	# Enhanced pathfinding: find multiple alternative paths with A* 
	def heuristic(a, b):
		return abs(a[0] - b[0]) + abs(a[1] - b[1])
	
	def find_path_astar(start_pos, end_pos):
		from heapq import heappush, heappop
		open_set = [(0, start_pos)]
		came_from = {}
		g_score = {start_pos: 0}
		f_score = {start_pos: heuristic(start_pos, end_pos)}
		
		while open_set:
			current = heappop(open_set)[1]
			
			if current == end_pos:
				path = []
				while current in came_from:
					r, c = current
					x = outer_rect.left + (c + 0.5) * cell_w
					y = outer_rect.top + (r + 0.5) * cell_h
					path.append((x, y))
					current = came_from[current]
				path.reverse()
				return path
			
			for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
				neighbor = (current[0] + dr, current[1] + dc)
				nr, nc = neighbor
				
				if not (0 <= nr < rows and 0 <= nc < cols):
					continue
				if grid[nr][nc]:  # wall
					continue
				
				tentative_g = g_score[current] + 1
				
				if neighbor not in g_score or tentative_g < g_score[neighbor]:
					came_from[neighbor] = current
					g_score[neighbor] = tentative_g
					f_score[neighbor] = tentative_g + heuristic(neighbor, end_pos)
					heappush(open_set, (f_score[neighbor], neighbor))
		
		return []
	
	# Find primary path using A*
	main_path = find_path_astar((start_r, start_c), (finish_r, finish_c))
	
	# Generate dense intermediate waypoints for smoother navigation
	if main_path:
		dense_path = []
		for i in range(len(main_path)):
			dense_path.append(main_path[i])
			# add intermediate points between waypoints for smoother curves
			if i < len(main_path) - 1:
				x1, y1 = main_path[i]
				x2, y2 = main_path[i + 1]
				# add 2-3 intermediate points
				for j in range(1, 4):
					frac = j / 4.0
					mid_x = x1 + (x2 - x1) * frac
					mid_y = y1 + (y2 - y1) * frac
					dense_path.append((mid_x, mid_y))
		path = dense_path
	else:
		# fallback to direct finish center if pathfinding failed
		path = [(finish_zone.centerx, finish_zone.centery)]

	# convert walls in grid into obstacle rects (fill cells where grid==True)
	obstacles = []
	for r in range(rows):
		for c in range(cols):
			if grid[r][c]:
				x = int(outer_rect.left + c*cell_w)
				y = int(outer_rect.top + r*cell_h)
				w = int(cell_w)
				h = int(cell_h)
				rect = pygame.Rect(x, y, w, h)
				# avoid placing wall in finish zone or start offset
				if not rect.colliderect(finish_zone):
					obstacles.append(rect)

	return obstacles, path

# --- DRIVER CLASS ---
class Driver:
	def __init__(self, name, color, start_pos, driver_id=0):
		self.name = name
		self.color = color
		self.x, self.y = start_pos
		self.angle = random.uniform(-0.2, 0.2)
		# speed is stored as pixels per second for smooth dt-based motion
		# start drivers slightly slower to avoid excessive top speeds
		self.speed = random.uniform(90.0, 140.0)
		# absolute max speed for this driver (used to cap aggressive bursts)
		self.max_speed = 180.0
		self.finished = False
		self.time = None
		self.start_time = time.time()
		self.id = driver_id
		# velocity components
		self.vx = math.cos(self.angle) * self.speed
		self.vy = math.sin(self.angle) * self.speed
		# drift-related
		self.drift = False
		self.drift_marks = []  # list of (x,y) points for tire marks
		self.max_marks = 200
		# drift control timers
		self.drift_timer = 0.0
		self.max_drift_time = 0.9
		self.drift_recovery = 3.0
		# recent bounce timer (seconds) - set when hitting wall/obstacle to allow short drift
		self.just_bounced = 0.0
		self.restitution = 0.6  # bounce restitution
		# waypoint following
		self.wp_index = 0
		self.wp_reached_dist = 35.0  # reduced for tighter following of dense waypoints
		self.max_turn_rate = 4.5  # increased for more responsive steering
		self.avoid_strength = 2.4
		# learning / adaptation: chance to successfully avoid obstacles
		self.avoid_success = 0.7
		self.learning_rate = 0.08
		self.last_collision_time = None
		# simple behavior state for higher-level tactics ('cruise','overtake','brake')
		self.state = 'cruise'
		self.state_timer = 0.0
		# personality/strategy: prefer -1 inner, 0 middle, 1 outer
		self.prefer = random.choice([-1, 0, 1])
		# aggressiveness scales target speed and risk-taking
		self.aggressiveness = random.uniform(0.8, 1.4)
		# caution reduces steering boldness
		self.caution = random.uniform(0.7, 1.2)
		# chance to pick an alternate route (overtake/avoid differently)
		self.alt_route_prob = random.uniform(0.05, 0.4)
		# angular velocity for smoothing steering
		self.ang_vel = 0.0
		# smoothing factor [0..1] (higher = smoother/slower reactions) - reduced for faster response
		self.steer_smooth = 0.18
		# base perception range for obstacle avoidance - increased for better planning
		self.perception_range = 260
		# enhanced pathfinding cache
		self.lookahead_waypoints = 3  # look ahead multiple waypoints for better planning

	def update(self, obstacles, dt, waypoints=None, drivers=None):
		if self.finished:
			return
		# waypoint following steering (higher-level AI) with enhanced lookahead
		steer = 0.0
		ang_step = 0.0
		if waypoints:
			# Enhanced waypoint following: look ahead multiple waypoints for smoother paths
			wp_index = self.wp_index % len(waypoints)
			
			# Lookahead strategy: find the furthest reachable waypoint within perception range
			target_wp_index = wp_index
			for i in range(min(self.lookahead_waypoints, len(waypoints) - wp_index)):
				check_index = (wp_index + i) % len(waypoints)
				check_wp = waypoints[check_index]
				dist_to_check = math.hypot(check_wp[0] - self.x, check_wp[1] - self.y)
				# if this waypoint is within reasonable lookahead distance, use it as target
				if dist_to_check < self.perception_range * 0.8:
					target_wp_index = check_index
				else:
					break
			
			wp = waypoints[target_wp_index]
			
			# personality: offset waypoint laterally based on prefer (-1 inner, 0 mid, 1 outer)
			# compute a small lateral offset perpendicular to direction to next waypoint
			dx = wp[0] - self.x
			dy = wp[1] - self.y
			base_ang = math.atan2(dy, dx)
			# lateral offset magnitude depends on TRACK_WIDTH
			# if this is the finish waypoint, aim directly at the center (no lateral offset)
			is_finish_wp = (target_wp_index == len(waypoints)-1)
			side_offset = 0.0 if is_finish_wp else (self.prefer * (TRACK_WIDTH/4.0))  # reduced offset for tighter racing line
			# sometimes pick an alternate small offset (overtake/alternate route)
			if random.random() < self.alt_route_prob:
				side_offset *= random.uniform(0.5, 1.2)
			# compute offset point
			lx = -math.sin(base_ang)
			ly = math.cos(base_ang)
			wp_x = wp[0] + lx * side_offset
			wp_y = wp[1] + ly * side_offset
			desired_ang = math.atan2(wp_y - self.y, wp_x - self.x)
			ang_diff = (desired_ang - self.angle + math.pi) % (2*math.pi) - math.pi
			# clamp turn rate
			max_turn = self.max_turn_rate * dt
			ang_step = max(-max_turn, min(max_turn, ang_diff))
			steer += ang_step
			# small randomness so they don't follow perfectly
			if random.random() < 0.08:  # reduced randomness for more precise following
				steer += random.uniform(-0.4, 0.4) * dt
			
			# Enhanced waypoint progression: check multiple nearby waypoints for completion
			# This prevents getting stuck on waypoints that are hard to reach precisely
			for check_offset in range(min(3, len(waypoints) - wp_index)):
				check_idx = (wp_index + check_offset) % len(waypoints)
				check_wp = waypoints[check_idx]
				check_dist = math.hypot(check_wp[0] - self.x, check_wp[1] - self.y)
				if check_dist < self.wp_reached_dist:
					# skip ahead to this waypoint
					self.wp_index = check_idx + 1
					break
		else:
			# fallback random jitter
			steer = random.uniform(-0.8, 0.8) * dt
			ang_step = steer
		# Enhanced speed adjustments with trajectory prediction
		# reduce base target so drivers are overall slower but smarter
		target_speed = 140.0 * self.aggressiveness
		
		# Cornering prediction: look ahead to anticipate turns and slow down early
		if waypoints and len(waypoints) > 1:
			current_wp_idx = self.wp_index % len(waypoints)
			# look ahead 2-3 waypoints to predict cornering needs
			lookahead_indices = []
			for offset in range(1, min(4, len(waypoints) - current_wp_idx)):
				lookahead_indices.append((current_wp_idx + offset) % len(waypoints))
			
			if lookahead_indices:
				# calculate required turning angles for upcoming waypoints
				total_turn_ahead = 0.0
				for i, wp_idx in enumerate(lookahead_indices):
					if wp_idx < len(waypoints) - 1:  # not the last waypoint
						curr_wp = waypoints[wp_idx]
						next_wp = waypoints[(wp_idx + 1) % len(waypoints)]
						prev_wp = waypoints[wp_idx - 1] if wp_idx > 0 else waypoints[wp_idx]
						
						# calculate the angle change between segments
						ang1 = math.atan2(curr_wp[1] - prev_wp[1], curr_wp[0] - prev_wp[0])
						ang2 = math.atan2(next_wp[1] - curr_wp[1], next_wp[0] - curr_wp[0])
						turn_angle = abs((ang2 - ang1 + math.pi) % (2*math.pi) - math.pi)
						
						# weight turns by proximity (closer turns matter more)
						weight = 1.0 / (i + 1)
						total_turn_ahead += turn_angle * weight
				
				# adjust speed based on upcoming turns
				turn_speed_factor = max(0.6, 1.0 - (total_turn_ahead / math.pi) * 0.4)
				target_speed *= turn_speed_factor
		
		# cornering slow-down: if steering large, reduce target speed
		steer_penalty = abs(steer) * (1.0 + (1.0 - self.caution)) * 70.0  # reduced penalty for smoother flow
		target_speed = max(70.0, target_speed - steer_penalty)
		
		# cap to driver's maximum speed
		target_speed = min(target_speed, getattr(self, 'max_speed', 200.0))
		
		# slow down when approaching finish zone for a safer final approach
		dist_to_finish = math.hypot(FINISH_ZONE.centerx - self.x, FINISH_ZONE.centery - self.y)
		if dist_to_finish < 300.0:
			approach_base = 60.0 + 30.0 * self.aggressiveness
			target_speed = min(target_speed, approach_base)
		
		# accelerate/decelerate toward target (gentler accel for smoother motion)
		accel = 120.0 * (0.6 + 0.6 * self.aggressiveness)
		if self.speed < target_speed:
			self.speed = min(target_speed, self.speed + accel * dt)
		else:
			self.speed = max(target_speed, self.speed - accel * dt)
		# DO NOT apply immediate small steer here; we'll compute a smoothed angular velocity
		# later after selecting a safe final heading. Keep steer value for diagnostics only.
		# forward vector
		fx = math.cos(self.angle)
		fy = math.sin(self.angle)
		# lateral (perp) vector
		lx = -fy
		ly = fx
		# determine drift intensity based on the last clamped ang_step
		# normalize turn magnitude per-frame to avoid large oscillations
		max_turn = self.max_turn_rate * dt
		drift_threshold = max(0.04, max_turn * 0.15)
		# normalized turn [0..1]
		normalized_turn = min(1.0, abs(ang_step) / (max_turn + 1e-6))
		drift_intensity = normalized_turn
		# require a moderate speed to drift to avoid low-speed circling
		# decay bounce timer
		self.just_bounced = max(0.0, getattr(self, 'just_bounced', 0.0) - dt)
		# trigger drift ONLY when recently bounced off wall/obstacle
		would_drift = (self.just_bounced > 0.0)
		# update drift timer and state (prevent endless drift by enforcing a max duration)
		if would_drift:
			self.drift_timer += dt
			self.drift = True
		else:
			# recover the timer faster when not drifting
			self.drift_timer = max(0.0, self.drift_timer - dt * self.drift_recovery)
			if self.drift_timer <= 0.001:
				self.drift = False
		# if drift lasted too long, force exit drift and reduce lateral tendency
		if self.drift_timer > self.max_drift_time:
			self.drift = False
			# clamp timer so it decays and we don't re-enter immediately
			self.drift_timer = self.max_drift_time * 0.6
		# base velocity along forward
		forward_vx = fx * self.speed
		forward_vy = fy * self.speed
		# sideways slip when drifting (dynamically reduced as drift persists)
		# very small baseline slip when not drifting to avoid visual sliding
		slip_base = drift_intensity * (0.45 if self.drift else 0.02)
		# reduce slip as drift_timer increases to make recovery more likely
		dynamic_slip = slip_base * max(0.22, 1.0 - (self.drift_timer / max(1e-6, self.max_drift_time)))
		lateral_vx = lx * self.speed * dynamic_slip
		lateral_vy = ly * self.speed * dynamic_slip
		# combine velocities
		self.vx = forward_vx + lateral_vx
		self.vy = forward_vy + lateral_vy
		# when drifting, gently align vehicle heading toward actual velocity vector to end drift
		if self.drift:
			vel_ang = math.atan2(self.vy, self.vx)
			# blend angle toward velocity angle; strength depends on dt and recovery param
			blend = min(0.8, 3.0 * dt)
			self.angle = (1.0 - blend) * self.angle + blend * vel_ang
		# integrate with dt
		new_x = self.x + self.vx * dt
		new_y = self.y + self.vy * dt
		# Smart avoidance: compute a repulsion vector field from nearby obstacles
		# and blend it with the waypoint-directed heading to produce a safe desired angle.
		lookahead = 120 + (self.speed / 3.0)
		# world-space repulsion vector
		rep_x, rep_y = 0.0, 0.0
		for obs in obstacles:
			# obstacle center
			ox = obs.x + obs.width/2
			oy = obs.y + obs.height/2
			# distance from driver to obstacle
			dx_o = self.x - ox
			dy_o = self.y - oy
			dist = math.hypot(dx_o, dy_o)
			influence = max(0.0, (self.perception_range + (self.speed*0.2) - dist))
			if influence > 0:
				# direction away from obstacle, stronger when closer and when projected path intersects
				away_x = dx_o / (dist + 1e-6)
				away_y = dy_o / (dist + 1e-6)
				# weight by obstacle size and closeness
				weight = (max(obs.width, obs.height)/40.0) * (influence / (self.perception_range + 1e-6))
				rep_x += away_x * weight
				rep_y += away_y * weight
		# also add a small lookahead collision check to bias strongly for obstacles ahead
		proj_x = self.x + fx * lookahead
		proj_y = self.y + fy * lookahead
		for obs in obstacles:
			ox = obs.x + obs.width/2
			oy = obs.y + obs.height/2
			dp = math.hypot(proj_x-ox, proj_y-oy)
			if dp < (max(obs.width, obs.height)/2 + 24 + self.speed*0.06):
				# push more strongly away from obstacles in projected path
				vx = proj_x - ox
				vy = proj_y - oy
				dpdist = math.hypot(vx, vy)
				if dpdist > 1e-6:
					rep_x += (vx/dpdist) * 1.6
					rep_y += (vy/dpdist) * 1.6
		# Inter-driver avoidance: add repulsion from nearby cars and simple overtake behavior
		if drivers:
			for other in drivers:
				if other is self or other.finished:
					continue
				dx_d = other.x - self.x
				dy_d = other.y - self.y
				dist_d = math.hypot(dx_d, dy_d)
				if dist_d <= 0:
					continue
				# general repulsion when quite close
				if dist_d < 150:
					push = (150.0 - dist_d) / 150.0
					rep_x -= (dx_d/dist_d) * push * 0.9
					rep_y -= (dy_d/dist_d) * push * 0.9
				# if the other car is roughly ahead, consider overtaking
				ang_to_other = math.atan2(dy_d, dx_d)
				rel_ang = (ang_to_other - self.angle + math.pi) % (2*math.pi) - math.pi
				if abs(rel_ang) < math.radians(55) and dist_d < 120:
					# try to overtake if faster
					if other.speed + 8.0 < self.speed:
						# bias a lateral repulsion to create a passing lane
						perp_x = -fy
						perp_y = fx
						side = -1.0 if random.random() < 0.5 else 1.0
						rep_x += perp_x * side * 0.8
						rep_y += perp_y * side * 0.8
						# temper speed slightly to avoid contact
						self.speed = max(self.speed * 0.92, 70.0)
					else:
						# if other is in side position and close, push lightly away
						if dist_d < 90:
							rep_x -= (dx_d/dist_d) * 0.5
							rep_y -= (dy_d/dist_d) * 0.5
		# compute desired angle from waypoint (maintain racing line preference)
		if waypoints:
			wp_index = self.wp_index % len(waypoints)
			wp = waypoints[wp_index]
			# same lateral offset usage to maintain racing lines, but not for finish waypoint
			dx = wp[0] - self.x
			dy = wp[1] - self.y
			base_ang = math.atan2(dy, dx)
			is_finish_wp = (wp_index == len(waypoints)-1)
			side_offset = 0.0 if is_finish_wp else (self.prefer * (TRACK_WIDTH/3.0))
			if not is_finish_wp and random.random() < self.alt_route_prob:
				side_offset *= random.uniform(0.7, 1.1)
			lx = -math.sin(base_ang)
			ly = math.cos(base_ang)
			wp_x = wp[0] + lx * side_offset
			wp_y = wp[1] + ly * side_offset
			desired_ang = math.atan2(wp_y - self.y, wp_x - self.x)
		else:
			desired_ang = math.atan2(fy, fx)

		# If repulsion exists, compute a blended desired direction vector
		rep_len = math.hypot(rep_x, rep_y)
		way_x = math.cos(desired_ang)
		way_y = math.sin(desired_ang)
		if rep_len > 1e-6:
			blend_factor = min(0.9, rep_len)
			bx = way_x - rep_x * blend_factor
			by = way_y - rep_y * blend_factor
			b_len = math.hypot(bx, by)
			if b_len > 1e-6:
				base_final_ang = math.atan2(by, bx)
			else:
				base_final_ang = desired_ang
		else:
			base_final_ang = desired_ang

		# --- Finish approach: if close to finish, force alignment and strong slow-down ---
		# distance to finish center
		dist_finish = math.hypot(FINISH_ZONE.centerx - self.x, FINISH_ZONE.centery - self.y)
		if dist_finish < 260.0:
			# force final angle directly toward finish center
			finish_ang = math.atan2(FINISH_ZONE.centery - self.y, FINISH_ZONE.centerx - self.x)
			# blend toward finish angle strongly
			base_final_ang = finish_ang
			# impose extra slow-down request by lowering speed target aggressively (handled below via dist_to_finish)

		# Predictive collision avoidance and simple state transitions
		# look a short time ahead to detect potential future conflicts
		predict_t = 0.6 + max(0.0, (self.speed-60.0)/200.0)
		pred_rep_x, pred_rep_y = 0.0, 0.0
		for other in drivers or []:
			if other is self or other.finished:
				continue
			# predict other future position assuming straight motion
			other_fx = math.cos(other.angle)
			other_fy = math.sin(other.angle)
			other_px = other.x + other_fx * other.speed * predict_t
			other_py = other.y + other_fy * other.speed * predict_t
			self_px = self.x + math.cos(self.angle) * self.speed * predict_t
			self_py = self.y + math.sin(self.angle) * self.speed * predict_t
			dx_p = self_px - other_px
			dy_p = self_py - other_py
			dist_p = math.hypot(dx_p, dy_p)
			if dist_p < 80.0:
				# strong repulsion away from predicted collision point
				pred_rep_x += dx_p / (dist_p + 1e-6) * (1.4 - dist_p/80.0)
				pred_rep_y += dy_p / (dist_p + 1e-6) * (1.4 - dist_p/80.0)
				# set an overtake state if the other is slower and roughly ahead
				ang_to_other = math.atan2(other.y - self.y, other.x - self.x)
				rel_ang = (ang_to_other - self.angle + math.pi) % (2*math.pi) - math.pi
				if other.speed + 6.0 < self.speed and abs(rel_ang) < math.radians(50):
					self.state = 'overtake'
					self.state_timer = 1.2
		# add predicted repulsion into rep vector
		rep_x += pred_rep_x * 1.3
		rep_y += pred_rep_y * 1.3

		# decay state timer
		if self.state_timer > 0:
			self.state_timer -= dt
		else:
			# return to cruise if timer expired
			if self.state == 'overtake':
				self.state = 'cruise'

		# Enhanced Candidate-steering lookahead: optimized sampling with adaptive resolution
		candidates = []
		# adaptive sampling: use more samples when in tight situations, fewer when clear
		nearby_obstacles = sum(1 for obs in obstacles 
		                      if math.hypot(obs.x + obs.width/2 - self.x, obs.y + obs.height/2 - self.y) < 150)
		nearby_drivers = sum(1 for other in (drivers or []) 
		                    if other is not self and not other.finished and math.hypot(other.x - self.x, other.y - self.y) < 120)
		
		# adaptive sample count based on local complexity
		if nearby_obstacles + nearby_drivers > 3:
			num_samples = 9  # more samples in crowded areas
		elif nearby_obstacles + nearby_drivers > 1:
			num_samples = 7  # moderate sampling
		else:
			num_samples = 5  # fewer samples when clear
		
		max_turn = self.max_turn_rate * dt * 1.0
		for i in range(num_samples):
			frac = (i / (num_samples-1)) if num_samples > 1 else 0.5
			cand_ang = self.angle + (frac*2.0-1.0) * max_turn
			# adaptive lookahead: shorter when turning, longer when straight
			base_look_t = 0.5 + (self.speed/400.0)
			turn_factor = abs(frac*2.0-1.0)  # 0 at center, 1 at extremes
			look_t = base_look_t * (1.0 - turn_factor * 0.3)  # reduce lookahead when turning hard
			
			proj_x = self.x + math.cos(cand_ang) * self.speed * look_t
			proj_y = self.y + math.sin(cand_ang) * self.speed * look_t
			
			# Fast clearance computation with early termination
			min_dist = 1e6
			collision_risk = False
			
			# check obstacles with spatial optimization
			for obs in obstacles:
				ox = obs.x + obs.width/2
				oy = obs.y + obs.height/2
				# quick distance check - skip distant obstacles
				quick_dist = max(abs(proj_x-ox), abs(proj_y-oy))
				if quick_dist > 200:
					continue
				d = math.hypot(proj_x-ox, proj_y-oy) - max(obs.width, obs.height)/2.0
				min_dist = min(min_dist, d)
				if d < 20:  # very close to collision
					collision_risk = True
					break
			
			if not collision_risk:
				# check other drivers only if no obstacle collision risk
				for other in drivers or []:
					if other is self or other.finished:
						continue
					# quick distance check
					quick_dist = max(abs(proj_x-other.x), abs(proj_y-other.y))
					if quick_dist > 120:
						continue
					d = math.hypot(proj_x-other.x, proj_y-other.y) - DRIVER_RADIUS*1.5
					min_dist = min(min_dist, d)
					if d < 25:
						collision_risk = True
						break
			
			# penalize going off-track (outside rectangular bounds)
			if proj_x < TRACK_MARGIN or proj_x > WIDTH-TRACK_MARGIN or proj_y < TRACK_MARGIN or proj_y > HEIGHT-TRACK_MARGIN:
				min_dist -= 100.0
				collision_risk = True
			
			# Early rejection for clearly bad candidates
			if collision_risk:
				candidates.append((-1000, cand_ang, min_dist))
				continue
			
			# alignment score: prefer headings that point toward base_final_ang
			align = math.cos(((cand_ang - base_final_ang + math.pi) % (2*math.pi)) - math.pi)
			
			# optimized overtake bonus computation
			overtake_bonus = 0.0
			if self.state == 'overtake':  # only compute when in overtake mode
				for other in drivers or []:
					if other is self or other.finished:
						continue
					dx_d = other.x - self.x
					dy_d = other.y - self.y
					dist_d = math.hypot(dx_d, dy_d)
					if dist_d < 140:
						ang_to_other = math.atan2(dy_d, dx_d)
						rel_ang = (ang_to_other - self.angle + math.pi) % (2*math.pi) - math.pi
						if abs(rel_ang) < math.radians(55) and other.speed + 6.0 < self.speed:
							# prefer candidates that create lateral separation from the slower car
							perp = -math.sin(self.angle)
							proj_lat = (math.cos(cand_ang)*perp*1.0 + math.sin(cand_ang)*(-perp)*1.0)
							overtake_bonus += 0.4 * max(0.0, proj_lat)
			
			# enhanced scoring: clearance (min_dist) + alignment*50 + overtake_bonus*10 + speed_bonus
			speed_bonus = 5.0 if min_dist > 60 else 0  # bonus for maintaining speed in clear areas
			score = min_dist + align*50.0 + overtake_bonus*10.0 + speed_bonus
			candidates.append((score, cand_ang, min_dist))
		
		# pick best candidate with fallback
		candidates.sort(key=lambda x: x[0], reverse=True)
		best_score, best_ang, best_clear = candidates[0]
		final_ang = best_ang

		# compute angular difference and apply smoothed angular velocity
		ang_diff = (final_ang - self.angle + math.pi) % (2*math.pi) - math.pi
		# target angular velocity proportional to ang_diff but clamped
		target_ang_vel = max(-self.max_turn_rate*1.2, min(self.max_turn_rate*1.2, ang_diff / max(0.01, dt)))
		# smooth the angular velocity
		self.ang_vel = (1.0 - self.steer_smooth) * target_ang_vel + self.steer_smooth * self.ang_vel
		# update angle by integrated angular velocity (scaled by dt)
		self.angle += self.ang_vel * dt
		# Wall collision
		if new_x < TRACK_MARGIN:
			new_x = TRACK_MARGIN + DRIVER_RADIUS
			# reflect velocity on vertical wall
			self.vx = -self.vx * self.restitution
			self.vy = self.vy * 0.9
			self.angle = math.atan2(self.vy, self.vx)
			self.just_bounced = 0.6
		if new_x > WIDTH-TRACK_MARGIN:
			new_x = WIDTH-TRACK_MARGIN - DRIVER_RADIUS
			self.vx = -self.vx * self.restitution
			self.vy = self.vy * 0.9
			self.angle = math.atan2(self.vy, self.vx)
			self.just_bounced = 0.6
		if new_y < TRACK_MARGIN:
			new_y = TRACK_MARGIN + DRIVER_RADIUS
			self.vy = -self.vy * self.restitution
			self.vx = self.vx * 0.9
			self.angle = math.atan2(self.vy, self.vx)
			self.just_bounced = 0.6
		if new_y > HEIGHT-TRACK_MARGIN:
			new_y = HEIGHT-TRACK_MARGIN - DRIVER_RADIUS
			self.vy = -self.vy * self.restitution
			self.vx = self.vx * 0.9
			self.angle = math.atan2(self.vy, self.vx)
			self.just_bounced = 0.6
		# Obstacle collision
		driver_rect = pygame.Rect(new_x-DRIVER_RADIUS, new_y-DRIVER_RADIUS, DRIVER_RADIUS*2, DRIVER_RADIUS*2)
		for obs in obstacles:
			if driver_rect.colliderect(obs):
				# compute normal from obstacle center to driver center
				ox = obs.x + obs.width/2
				oy = obs.y + obs.height/2
				dx = new_x - ox
				dy = new_y - oy
				dist = math.hypot(dx, dy)
				if dist == 0:
					nx, ny = 1.0, 0.0
				else:
					nx, ny = dx/dist, dy/dist
				# reflect velocity about normal: v' = v - 2*(v·n)*n
				vdotn = self.vx*nx + self.vy*ny
				self.vx = self.vx - 2*vdotn*nx
				self.vy = self.vy - 2*vdotn*ny
				# apply restitution and slight random spin
				self.vx *= self.restitution
				self.vy *= self.restitution
				self.angle = math.atan2(self.vy, self.vx)
				# mark recent bounce so drift is allowed briefly after collision
				self.just_bounced = 0.6
				# learning: collision reduces avoid_success for a while (they learn from mistakes slowly)
				self.avoid_success = max(0.2, self.avoid_success - 0.15)
				self.last_collision_time = time.time()
				# push driver out of obstacle a bit
				overlap_push = DRIVER_RADIUS + max(obs.width, obs.height)/2 - dist
				if overlap_push > 0:
					new_x += nx * overlap_push
					new_y += ny * overlap_push
				break
		self.x, self.y = new_x, new_y
		# decay / recover avoid_success slowly over time
		if self.last_collision_time:
			recover_t = time.time() - self.last_collision_time
			if recover_t > 1.0:
				self.avoid_success = min(0.95, self.avoid_success + self.learning_rate * min(1.0, recover_t/5.0))
		# Finish check - simple finish when entering zone
		if FINISH_ZONE.collidepoint(self.x, self.y) and not self.finished:
			self.finished = True
			self.time = time.time() - self.start_time

	def draw(self, surface):
		# Draw the car as a circle with a little direction line
		cx, cy = int(self.x), int(self.y)
		pygame.draw.circle(surface, self.color, (cx, cy), DRIVER_RADIUS)
		# direction indicator
		dx = int(math.cos(self.angle) * DRIVER_RADIUS * 1.6)
		dy = int(math.sin(self.angle) * DRIVER_RADIUS * 1.6)
		pygame.draw.line(surface, (0,0,0), (cx, cy), (cx+dx, cy+dy), 2)
		name_surf = font.render(self.name, True, (0,0,0))
		surface.blit(name_surf, (cx-name_surf.get_width()//2, cy-DRIVER_RADIUS-20))
		# add tire mark point if drifting (record timestamped mark)
		if self.drift:
			marks_list.append({
				'id': self.id,
				'pos': (self.x, self.y),
				't': time.time(),
				'color': (20,20,20)
			})

# --- MAIN ---
def main():
	# define finish zone relative to current fullscreen size
	global FINISH_ZONE
	# make finish area smaller so drivers must aim more precisely
	FINISH_ZONE = pygame.Rect(WIDTH-160, HEIGHT//2-50, 100, 100)
	# create rectangular track: outer rect and inner rect
	outer_rect = pygame.Rect(TRACK_MARGIN, TRACK_MARGIN, WIDTH-2*TRACK_MARGIN, HEIGHT-2*TRACK_MARGIN)
	inner_rect = pygame.Rect(TRACK_MARGIN+TRACK_WIDTH, TRACK_MARGIN+TRACK_WIDTH, WIDTH-2*(TRACK_MARGIN+TRACK_WIDTH), HEIGHT-2*(TRACK_MARGIN+TRACK_WIDTH))
	# generate waypoints along the mid-line of the rectangular track (clockwise)
	waypoints = []
	# top edge (left->right)
	mid_top = TRACK_MARGIN + TRACK_WIDTH/2
	waypoints.append((outer_rect.left + TRACK_WIDTH/2, outer_rect.top + TRACK_WIDTH/2))
	waypoints.append((outer_rect.right - TRACK_WIDTH/2, outer_rect.top + TRACK_WIDTH/2))
	waypoints.append((outer_rect.right - TRACK_WIDTH/2, outer_rect.bottom - TRACK_WIDTH/2))
	waypoints.append((outer_rect.left + TRACK_WIDTH/2, outer_rect.bottom - TRACK_WIDTH/2))
	# add an explicit waypoint at the finish center to guide drivers precisely
	waypoints.append((FINISH_ZONE.centerx, FINISH_ZONE.centery))
	# generate obstacles constrained to track area
	obstacles = generate_obstacles(FINISH_ZONE, inner_rect, outer_rect)
	# start positions: place drivers staggered near left-middle of track inside lane
	start_x = outer_rect.left + TRACK_WIDTH/2 + 20
	start_y = HEIGHT//2 - (DRIVER_COUNT-1)*18
	start_positions = [(start_x, start_y + i*36) for i in range(DRIVER_COUNT)]
	# load learning and create drivers with persisted avoid_success when available
	learning_data = load_learning()
	drivers = []
	for i in range(DRIVER_COUNT):
		p = Driver(DRIVER_NAMES[i], DRIVER_COLORS[i], start_positions[i], driver_id=i)
		key = f"driver_{i}"
		if key in learning_data:
			p.avoid_success = learning_data[key].get('avoid_success', p.avoid_success)
			drivers.append(p)
		else:
			drivers.append(p)
	running = True
	sim_start = time.time()
	# perform an initial flip to ensure the display buffer is initialized
	pygame.display.flip()
	last_time = time.time()
	while running:
		# pump early to ensure the window system can update the display state
		pygame.event.pump()
		now = time.time()
		dt = min(1.0/30.0, now - last_time)
		last_time = now
		# Update and draw drivers first (they append marks to marks_list)
		for driver in drivers:
			if not driver.finished and time.time() - driver.start_time < MAX_TIME:
				driver.update(obstacles, dt, waypoints, drivers)
		# Remove expired marks from marks_list and draw remaining with fading alpha
		nowt = time.time()
		# clear track_surface and redraw marks each frame to enable fade-out
		track_surface.fill((0,0,0,0))
		new_marks = []
		for m in marks_list:
			age = nowt - m['t']
			if age <= MARK_LIFETIME:
				alpha = int(255 * (1 - age / MARK_LIFETIME))
				color = m.get('color', (20,20,20)) + (alpha,)
				x, y = int(m['pos'][0]), int(m['pos'][1])
				pygame.draw.circle(track_surface, color, (x,y), 3)
				new_marks.append(m)
		marks_list[:] = new_marks
		# Now render scene
		screen.fill((180,200,180))
		# show persistent tire marks
		screen.blit(track_surface, (0,0))
		# Draw track border
		pygame.draw.rect(screen, (0,0,0), (TRACK_MARGIN, TRACK_MARGIN, WIDTH-2*TRACK_MARGIN, HEIGHT-2*TRACK_MARGIN), 4)
		# Draw finish zone
		pygame.draw.rect(screen, (0,255,0), FINISH_ZONE)
		finish_text = font.render("FINISH", True, (0,0,0))
		screen.blit(finish_text, (FINISH_ZONE.x+5, FINISH_ZONE.y+FINISH_ZONE.height//2-10))
		# Draw obstacles
		for obs in obstacles:
			pygame.draw.rect(screen, (0,0,0), obs)
		# Draw drivers (after marks so cars are on top)
		for driver in drivers:
			driver.draw(screen)
		# Show times
		for i, driver in enumerate(drivers):
			t = driver.time if driver.finished else (time.time()-driver.start_time if time.time()-driver.start_time < MAX_TIME else None)
			t_str = f"{t:.1f}s" if t is not None else "-"
			status = "FINISHED" if driver.finished else "RACING"
			txt = font.render(f"{driver.name}: {t_str} [{status}]", True, driver.color)
			screen.blit(txt, (10, 10 + i*22))

		# simple leaderboard: sort by finished first, then by time
		board = sorted(drivers, key=lambda d: (not d.finished, d.time if d.finished else 1e6))
		lb_x = WIDTH - 300
		screen.blit(font.render("Leaderboard:", True, (0,0,0)), (lb_x, 10))
		for i, d in enumerate(board[:6]):
			status = "FINISHED" if d.finished else "RACING"
			time_str = f"{d.time:.1f}s" if d.finished else "-"
			line = font.render(f"{i+1}. {d.name} {time_str} [{status}]", True, d.color)
			screen.blit(line, (lb_x, 34 + i*20))

		# End after all finished or time up, or when one driver remains
		alive = [d for d in drivers if not d.finished and (time.time()-d.start_time) < MAX_TIME]
		if len(alive) <= 1 or all(d.finished or time.time()-d.start_time >= MAX_TIME for d in drivers):
			end_text = font.render("Simulation finished. Press any key to exit.", True, (0,0,0))
			screen.blit(end_text, (WIDTH//2-180, HEIGHT-40))
			pygame.display.flip()
			wait = True
			while wait:
				for event in pygame.event.get():
					if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
						wait = False
						running = False
				clock.tick(30)
			break
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
		pygame.display.flip()
		# cap to FPS
		clock.tick(FPS)
	pygame.quit()
	# save learning on exit
	ldata = {}
	for d in drivers:
		ldata[f"driver_{d.id}"] = {'avoid_success': d.avoid_success}
	save_learning(ldata)

if __name__ == "__main__":
	main()
