# Race Track Simulation with Drivers
import pygame
import random
import math
import time

# --- CONFIG ---
WIDTH, HEIGHT = 900, 600
TRACK_MARGIN = 60
OBSTACLE_COUNT = 12
DRIVER_COUNT = 10
MAX_TIME = 60
DRIVER_RADIUS = 12
FINISH_ZONE = pygame.Rect(WIDTH-100, HEIGHT//2-60, 60, 120)
DRIVER_COLORS = [(255,0,0),(0,0,255),(255,128,0),(128,0,255),(0,128,255),(0,200,0),(200,200,0),(255,0,200),(0,200,200),(128,128,128)]
DRIVER_NAMES = [f"Driver {i+1}" for i in range(DRIVER_COUNT)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Race Track Simulation")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# --- OBSTACLES ---
def generate_obstacles():
	obstacles = []
	for _ in range(OBSTACLE_COUNT):
		while True:
			x = random.randint(TRACK_MARGIN, WIDTH-TRACK_MARGIN-60)
			y = random.randint(TRACK_MARGIN, HEIGHT-TRACK_MARGIN-60)
			rect = pygame.Rect(x, y, random.randint(40, 80), random.randint(40, 80))
			if not rect.colliderect(FINISH_ZONE):
				obstacles.append(rect)
				break
	return obstacles

# --- DRIVER CLASS ---
class Driver:
	def __init__(self, name, color, start_pos):
		self.name = name
		self.color = color
		self.x, self.y = start_pos
		self.angle = random.uniform(-0.2, 0.2)
		self.speed = random.uniform(2.0, 3.0)
		self.finished = False
		self.time = None
		self.start_time = time.time()
		self.vx = math.cos(self.angle) * self.speed
		self.vy = math.sin(self.angle) * self.speed

	def update(self, obstacles):
		if self.finished:
			return
		# Randomly adjust direction a bit
		self.angle += random.uniform(-0.15, 0.15)
		self.speed = min(4.0, max(1.5, self.speed + random.uniform(-0.1, 0.1)))
		self.vx = math.cos(self.angle) * self.speed
		self.vy = math.sin(self.angle) * self.speed
		new_x = self.x + self.vx
		new_y = self.y + self.vy
		# Wall collision
		if new_x < TRACK_MARGIN or new_x > WIDTH-TRACK_MARGIN:
			self.angle = math.pi - self.angle
			new_x = self.x
		if new_y < TRACK_MARGIN or new_y > HEIGHT-TRACK_MARGIN:
			self.angle = -self.angle
			new_y = self.y
		# Obstacle collision
		driver_rect = pygame.Rect(new_x-DRIVER_RADIUS, new_y-DRIVER_RADIUS, DRIVER_RADIUS*2, DRIVER_RADIUS*2)
		for obs in obstacles:
			if driver_rect.colliderect(obs):
				self.angle += math.pi/2 + random.uniform(-0.5,0.5)
				return
		self.x, self.y = new_x, new_y
		# Finish check
		if FINISH_ZONE.collidepoint(self.x, self.y):
			self.finished = True
			self.time = time.time() - self.start_time

	def draw(self, surface):
		pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), DRIVER_RADIUS)
		name_surf = font.render(self.name, True, (0,0,0))
		surface.blit(name_surf, (int(self.x)-name_surf.get_width()//2, int(self.y)-DRIVER_RADIUS-18))

# --- MAIN ---
def main():
	obstacles = generate_obstacles()
	start_positions = [(TRACK_MARGIN+30, HEIGHT//2 + i*30 - DRIVER_COUNT*15) for i in range(DRIVER_COUNT)]
	drivers = [Driver(DRIVER_NAMES[i], DRIVER_COLORS[i], start_positions[i]) for i in range(DRIVER_COUNT)]
	running = True
	sim_start = time.time()
	while running:
		screen.fill((255,255,255))
		# Draw track border
		pygame.draw.rect(screen, (0,0,0), (TRACK_MARGIN, TRACK_MARGIN, WIDTH-2*TRACK_MARGIN, HEIGHT-2*TRACK_MARGIN), 4)
		# Draw finish zone
		pygame.draw.rect(screen, (0,255,0), FINISH_ZONE)
		finish_text = font.render("FINISH", True, (0,0,0))
		screen.blit(finish_text, (FINISH_ZONE.x+5, FINISH_ZONE.y+FINISH_ZONE.height//2-10))
		# Draw obstacles
		for obs in obstacles:
			pygame.draw.rect(screen, (0,0,0), obs)
		# Update and draw drivers
		for driver in drivers:
			if not driver.finished and time.time() - driver.start_time < MAX_TIME:
				driver.update(obstacles)
			driver.draw(screen)
		# Show times
		for i, driver in enumerate(drivers):
			t = driver.time if driver.finished else (time.time()-driver.start_time if time.time()-driver.start_time < MAX_TIME else None)
			t_str = f"{t:.1f}s" if t is not None else "-"
			txt = font.render(f"{driver.name}: {t_str}", True, driver.color)
			screen.blit(txt, (10, 10 + i*22))
		# End after all finished or time up
		if all(d.finished or time.time()-d.start_time >= MAX_TIME for d in drivers):
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
		pygame.display.flip()
		clock.tick(60)
	pygame.quit()

if __name__ == "__main__":
	main()
