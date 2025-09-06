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

# --- DRIVER CLASS ---
class Driver:
	def __init__(self, name, color, start_pos, driver_id=0):
		self.name = name
		self.color = color
		self.x, self.y = start_pos
		self.angle = random.uniform(-0.2, 0.2)
		# speed is stored as pixels per second for smooth dt-based motion
		self.speed = random.uniform(120.0, 180.0)
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
		self.restitution = 0.6  # bounce restitution
		# waypoint following
		self.wp_index = 0
		self.wp_reached_dist = 40.0
		self.max_turn_rate = 3.0  # radians/sec max turn
		self.avoid_strength = 2.4
		# learning / adaptation: chance to successfully avoid obstacles
		self.avoid_success = 0.7
		self.learning_rate = 0.08
		self.last_collision_time = None
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
		# smoothing factor [0..1] (higher = smoother/slower reactions)
		self.steer_smooth = 0.28
		# base perception range for obstacle avoidance
		self.perception_range = 200

	def update(self, obstacles, dt, waypoints=None):
		if self.finished:
			return
		# waypoint following steering (higher-level AI)
		steer = 0.0
		ang_step = 0.0
		if waypoints:
			wp = waypoints[self.wp_index % len(waypoints)]
			# personality: offset waypoint laterally based on prefer (-1 inner, 0 mid, 1 outer)
			# compute a small lateral offset perpendicular to direction to next waypoint
			dx = wp[0] - self.x
			dy = wp[1] - self.y
			base_ang = math.atan2(dy, dx)
			# lateral offset magnitude depends on TRACK_WIDTH
			side_offset = (self.prefer * (TRACK_WIDTH/3.0))
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
			if random.random() < 0.12:
				steer += random.uniform(-0.6, 0.6) * dt
			# check waypoint reached (use offset target distance)
			if math.hypot(wp_x-self.x, wp_y-self.y) < self.wp_reached_dist:
				# sometimes skip to next waypoint if aggressive
				if random.random() < min(0.5, self.aggressiveness-0.6):
					self.wp_index += 2
				else:
					self.wp_index += 1
		else:
			# fallback random jitter
			steer = random.uniform(-0.8, 0.8) * dt
			ang_step = steer
		# speed adjustments (px/s) influenced by aggressiveness and caution
		target_speed = 160.0 * self.aggressiveness
		# cornering slow-down: if steering large, reduce target speed
		steer_penalty = abs(steer) * (1.0 + (1.0 - self.caution)) * 80.0
		target_speed = max(90.0, target_speed - steer_penalty)
		# accelerate/decelerate toward target
		accel = 200.0 * (0.6 + 0.8 * self.aggressiveness)
		if self.speed < target_speed:
			self.speed = min(target_speed, self.speed + accel * dt)
		else:
			self.speed = max(target_speed, self.speed - accel * dt)
		self.angle += steer
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
		# require a bit higher speed to drift to avoid low-speed circling
		self.drift = (abs(ang_step) > drift_threshold and self.speed > 150.0)
		# base velocity along forward
		forward_vx = fx * self.speed
		forward_vy = fy * self.speed
		# sideways slip when drifting (reduced to prevent circling)
		slip = drift_intensity * (0.45 if self.drift else 0.08)
		lateral_vx = lx * self.speed * slip
		lateral_vy = ly * self.speed * slip
		# combine velocities
		self.vx = forward_vx + lateral_vx
		self.vy = forward_vy + lateral_vy
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
		# combine waypoint desired angle with repulsion
		# compute desired angle from waypoint as before
		if waypoints:
			wp = waypoints[self.wp_index % len(waypoints)]
			# same lateral offset usage to maintain racing lines
			dx = wp[0] - self.x
			dy = wp[1] - self.y
			base_ang = math.atan2(dy, dx)
			side_offset = (self.prefer * (TRACK_WIDTH/3.0))
			if random.random() < self.alt_route_prob:
				side_offset *= random.uniform(0.7, 1.1)
			lx = -math.sin(base_ang)
			ly = math.cos(base_ang)
			wp_x = wp[0] + lx * side_offset
			wp_y = wp[1] + ly * side_offset
			desired_ang = math.atan2(wp_y - self.y, wp_x - self.x)
		else:
			desired_ang = math.atan2(fy, fx)
		# if repulsion is present, adjust desired heading away from rep vector
		rep_len = math.hypot(rep_x, rep_y)
		if rep_len > 1e-6:
			# form a blended desired vector
			blend_factor = min(0.92, min(0.9, rep_len))
			# waypoint direction vector
			way_x = math.cos(desired_ang)
			way_y = math.sin(desired_ang)
			# blended vector = normalize(way + (-rep)*blend_factor)
			bx = way_x - rep_x * blend_factor
			by = way_y - rep_y * blend_factor
			b_len = math.hypot(bx, by)
			if b_len > 1e-6:
				final_ang = math.atan2(by, bx)
			else:
				final_ang = desired_ang
		else:
			final_ang = desired_ang
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
		if new_x > WIDTH-TRACK_MARGIN:
			new_x = WIDTH-TRACK_MARGIN - DRIVER_RADIUS
			self.vx = -self.vx * self.restitution
			self.vy = self.vy * 0.9
			self.angle = math.atan2(self.vy, self.vx)
		if new_y < TRACK_MARGIN:
			new_y = TRACK_MARGIN + DRIVER_RADIUS
			self.vy = -self.vy * self.restitution
			self.vx = self.vx * 0.9
			self.angle = math.atan2(self.vy, self.vx)
		if new_y > HEIGHT-TRACK_MARGIN:
			new_y = HEIGHT-TRACK_MARGIN - DRIVER_RADIUS
			self.vy = -self.vy * self.restitution
			self.vx = self.vx * 0.9
			self.angle = math.atan2(self.vy, self.vx)
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
		# Finish check
		if FINISH_ZONE.collidepoint(self.x, self.y):
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
	FINISH_ZONE = pygame.Rect(WIDTH-200, HEIGHT//2-80, 140, 160)
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
				driver.update(obstacles, dt, waypoints)
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
			txt = font.render(f"{driver.name}: {t_str}", True, driver.color)
			screen.blit(txt, (10, 10 + i*22))

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
