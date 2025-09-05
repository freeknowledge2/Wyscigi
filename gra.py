# Race Track Simulation with Drivers
import pygame
import random
import math
import time

# --- CONFIG ---
# default (overridden on init)
WIDTH, HEIGHT = 900, 600  # will be overridden to fullscreen after init
TRACK_MARGIN = 80
OBSTACLE_COUNT = 14
DRIVER_COUNT = 10
MAX_TIME = 60
DRIVER_RADIUS = 12
FPS = 60
# how long a tire mark stays on track (seconds)
MARK_LIFETIME = 3.0
# placeholder; will be set after fullscreen init
FINISH_ZONE = pygame.Rect(0, 0, 0, 0)
DRIVER_COLORS = [(255,0,0),(0,0,255),(255,128,0),(128,0,255),(0,128,255),(0,200,0),(200,200,0),(255,0,200),(0,200,200),(128,128,128)]
DRIVER_NAMES = [f"Driver {i+1}" for i in range(DRIVER_COUNT)]

pygame.init()
# run in windowed mode for easier testing; set to False to use fullscreen
WINDOWED = True
# default window size when WINDOWED=True
WINDOW_SIZE = (1280, 720)
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

# --- OBSTACLES ---
def generate_obstacles(finish_zone):
	obstacles = []
	for _ in range(OBSTACLE_COUNT):
		attempts = 0
		while True:
			attempts += 1
			x = random.randint(TRACK_MARGIN, WIDTH-TRACK_MARGIN-80)
			y = random.randint(TRACK_MARGIN, HEIGHT-TRACK_MARGIN-80)
			rect = pygame.Rect(x, y, random.randint(50, 110), random.randint(40, 90))
			if not rect.colliderect(finish_zone):
				obstacles.append(rect)
				break
			if attempts > 20:
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

	def update(self, obstacles, dt):
		if self.finished:
			return
		# steering: random jitter but sometimes stronger to create drifts
		# scale steering by dt to be frame-rate independent
		steer = random.uniform(-1.2, 1.2) * dt
		if random.random() < 0.08:
			steer *= random.uniform(2.0, 4.0)
		# speed adjustments (px/s)
		self.speed = min(300.0, max(80.0, self.speed + random.uniform(-60.0, 60.0) * dt))
		self.angle += steer
		# forward vector
		fx = math.cos(self.angle)
		fy = math.sin(self.angle)
		# lateral (perp) vector
		lx = -fy
		ly = fx
		# determine drift intensity
		drift_threshold = 0.6 * dt
		drift_intensity = min(1.0, max(0.0, abs(steer) / (1.2 * dt + 1e-6)))
		# speed threshold in px/s approximate to old value
		self.drift = (abs(steer) > drift_threshold and self.speed > 140.0)
		# base velocity along forward
		forward_vx = fx * self.speed
		forward_vy = fy * self.speed
		# sideways slip when drifting
		slip = drift_intensity * (0.6 if self.drift else 0.15)
		lateral_vx = lx * self.speed * slip
		lateral_vy = ly * self.speed * slip
		# combine velocities
		self.vx = forward_vx + lateral_vx
		self.vy = forward_vy + lateral_vy
		# integrate with dt
		new_x = self.x + self.vx * dt
		new_y = self.y + self.vy * dt
		# obstacle avoidance AI: lookahead and try to steer away (not perfect)
		lookahead = 160 + (self.speed / 2.0)
		avoid_steer = 0.0
		avoid_success = 0.7
		for obs in obstacles:
			# distance from projected point ahead to obstacle center
			proj_x = self.x + fx * lookahead
			proj_y = self.y + fy * lookahead
			ox = obs.x + obs.width/2
			oy = obs.y + obs.height/2
			dist_to_proj = math.hypot(proj_x-ox, proj_y-oy)
			danger_radius = max(obs.width, obs.height) / 2 + 20
			if dist_to_proj < danger_radius + (self.speed/8.0):
				# compute angle to obstacle and steer away
				ang_to_obs = math.atan2(oy - self.y, ox - self.x)
				# desired steer is away from obstacle
				ang_diff = (ang_to_obs - self.angle + math.pi) % (2*math.pi) - math.pi
				# steer sign opposite of ang_diff
				desired = -0.8 * math.copysign(1.0, ang_diff) * dt
				# sometimes fail to avoid perfectly
				if random.random() < avoid_success:
					avoid_steer += desired
				else:
					avoid_steer += desired * random.uniform(0.2, 0.6)
		# apply avoidance (blend with random steer)
		self.angle += avoid_steer
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
				# push driver out of obstacle a bit
				overlap_push = DRIVER_RADIUS + max(obs.width, obs.height)/2 - dist
				if overlap_push > 0:
					new_x += nx * overlap_push
					new_y += ny * overlap_push
				break
		self.x, self.y = new_x, new_y
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
	obstacles = generate_obstacles(FINISH_ZONE)
	start_positions = [(TRACK_MARGIN+40, HEIGHT//2 + i*36 - DRIVER_COUNT*18) for i in range(DRIVER_COUNT)]
	drivers = [Driver(DRIVER_NAMES[i], DRIVER_COLORS[i], start_positions[i], driver_id=i) for i in range(DRIVER_COUNT)]
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
				driver.update(obstacles, dt)
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

if __name__ == "__main__":
	main()
