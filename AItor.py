import pygame
import numpy as np
import math
import random
from typing import List, Tuple
from dataclasses import dataclass

# Konfiguracja gry
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
PRETRAIN_GENERATIONS = 300  # liczba generacji szybkiego pre-treningu na starcie
PRETRAIN_SHOW_PROGRESS = True  # pokazuj postęp na ekranie podczas pre-treningu
PRETRAIN_EVENT_INTERVAL = 1    # co ile generacji pompować zdarzenia (1 = każda)
PRETRAIN_DRAW_INTERVAL = 5     # co ile generacji odświeżyć ekran (tekst postępu)

# Nowe parametry rozgrywki
ROUND_DURATION_SECONDS = 5      # długość jednej generacji w sekundach
MIN_SPEED = 0.8                 # minimalna prędkość (brak pełnego zatrzymania)
START_INITIAL_SPEED = 1.5       # początkowa prędkość startowa
START_BOOST_FRAMES = 40         # ile pierwszych klatek wymuszać ruch do przodu

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
GRASS_GREEN = (34, 139, 34)
TRACK_GRAY = (105, 105, 105)

# Kolory dla 10 agentów
AGENT_COLORS = [
    (255, 0, 0),    # Czerwony
    (0, 255, 0),    # Zielony
    (0, 0, 255),    # Niebieski
    (255, 255, 0),  # Żółty
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Pomarańczowy
    (128, 0, 128),  # Fioletowy
    (255, 192, 203), # Różowy
    (165, 42, 42)   # Brązowy
]

@dataclass
class Point:
    x: float
    y: float

class NeuralNetwork:
    """Prosta sieć neuronowa dla sterowania agentem"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 16, output_size: int = 3):
        # Losowa inicjalizacja wag
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Przejście w przód przez sieć"""
        # Pierwsza warstwa z aktywacją ReLU
        z1 = np.dot(inputs, self.w1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Druga warstwa z aktywacją tanh
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.tanh(z2)
        
        return a2
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3):
        """Mutacja sieci dla algorytmu ewolucyjnego"""
        if np.random.random() < mutation_rate:
            self.w1 += np.random.randn(*self.w1.shape) * mutation_strength
        if np.random.random() < mutation_rate:
            self.b1 += np.random.randn(*self.b1.shape) * mutation_strength
        if np.random.random() < mutation_rate:
            self.w2 += np.random.randn(*self.w2.shape) * mutation_strength
        if np.random.random() < mutation_rate:
            self.b2 += np.random.randn(*self.b2.shape) * mutation_strength
    
    def copy(self):
        """Kopia sieci"""
        new_net = NeuralNetwork()
        new_net.w1 = self.w1.copy()
        new_net.b1 = self.b1.copy()
        new_net.w2 = self.w2.copy()
        new_net.b2 = self.b2.copy()
        return new_net

class Car:
    """Klasa reprezentująca pojazd wyścigowy"""

    def __init__(self, x: float, y: float, angle: float, agent_id: int):
        # Pozycja i orientacja
        self.x = x
        self.y = y
        self.angle = angle

        # Fizyka
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed = START_INITIAL_SPEED  # początkowy impuls
        self.max_speed = 8.0
        self.acceleration = 0.35
        self.friction = 0.96
        self.turn_speed = 0.08

        # Start boost
        self.start_frames = 0

        # Wymiary
        self.width = 12
        self.height = 24

        # Stan gry
        self.alive = True
        self.score = 0.0
        self.distance_traveled = 0.0
        self.last_x = x
        self.last_y = y
        self.agent_id = agent_id
        self.color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]

        # Sensory
        self.sensor_angles = [i * (2 * math.pi / 8) for i in range(8)]
        self.sensor_distances = [1.0] * 8
        self.sensor_range = 100.0

        # Mózg
        self.brain = NeuralNetwork()

    # ---------------- Sensors ----------------
    def update_sensors(self, track):
        for i, sensor_angle in enumerate(self.sensor_angles):
            absolute_angle = self.angle + sensor_angle
            min_distance = self.sensor_range
            step_size = 4
            cos_a = math.cos(absolute_angle)
            sin_a = math.sin(absolute_angle)
            for step in range(step_size, int(self.sensor_range) + 1, step_size):
                sx = self.x + step * cos_a
                sy = self.y + step * sin_a
                if not track.is_on_track(sx, sy):
                    min_distance = step
                    break
            self.sensor_distances[i] = min_distance / self.sensor_range

    def get_neural_input(self) -> np.ndarray:
        return np.array(self.sensor_distances)

    # ---------------- Control & Update ----------------
    def update(self, track):
        if not self.alive:
            return

        self.update_sensors(track)
        output = self.brain.forward(self.get_neural_input())
        accel_out, left_out, right_out = output

        # Start boost wymusza ruch naprzód niezależnie od sieci
        if self.start_frames < START_BOOST_FRAMES:
            self.speed = max(self.speed, START_INITIAL_SPEED)
            self.accelerate()  # dodatkowe przyspieszenie
            self.start_frames += 1
        else:
            if accel_out > 0.2:
                self.accelerate()
            elif accel_out < -0.25:
                self.brake()

        if left_out > 0.3:
            self.turn_left()
        elif right_out > 0.3:
            self.turn_right()

        self.apply_physics()

        if not track.is_on_track(self.x, self.y):
            if self.alive:
                self.alive = False
                self.score -= 100

        self.update_score()

    def accelerate(self):
        self.speed = min(self.speed + self.acceleration, self.max_speed)

    def brake(self):
        self.speed = max(self.speed - self.acceleration * 2, MIN_SPEED)

    def turn_left(self):
        if self.speed > MIN_SPEED:
            self.angle -= self.turn_speed * (self.speed / self.max_speed)

    def turn_right(self):
        if self.speed > MIN_SPEED:
            self.angle += self.turn_speed * (self.speed / self.max_speed)

    def apply_physics(self):
        self.velocity_x = self.speed * math.cos(self.angle)
        self.velocity_y = self.speed * math.sin(self.angle)
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.speed *= self.friction
        if self.speed < MIN_SPEED:
            self.speed = MIN_SPEED

    def update_score(self):
        dist = math.hypot(self.x - self.last_x, self.y - self.last_y)
        if dist > 0.1:
            self.distance_traveled += dist
            self.score += dist * 0.1
        else:
            self.score -= 0.01
        self.last_x = self.x
        self.last_y = self.y

    def draw(self, screen: pygame.Surface):
        color = self.color if self.alive else (self.color[0]//3, self.color[1]//3, self.color[2]//3)
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        pts = [
            (-self.width//2, -self.height//2),
            (self.width//2, -self.height//2),
            (self.width//2, self.height//2),
            (-self.width//2, self.height//2)
        ]
        rotated = []
        for px, py in pts:
            rx = px * cos_a - py * sin_a + self.x
            ry = px * sin_a + py * cos_a + self.y
            rotated.append((rx, ry))
        pygame.draw.polygon(screen, color, rotated)
        if self.alive:
            fx = self.x + (self.height//2) * cos_a
            fy = self.y + (self.height//2) * sin_a
            pygame.draw.circle(screen, WHITE, (int(fx), int(fy)), 3)
        else:
            xsz = 8
            pygame.draw.line(screen, RED, (self.x - xsz, self.y - xsz), (self.x + xsz, self.y + xsz), 2)
            pygame.draw.line(screen, RED, (self.x + xsz, self.y - xsz), (self.x - xsz, self.y + xsz), 2)

class Track:
    """Klasa reprezentująca tor wyścigowy"""

    def __init__(self):
        # Parametry
        self.base_track_width = 70  # pół-szerokość toru (od środka do krawędzi)
        self.control_points = self._create_control_points()
        self.center_points = self._interpolate_centerline(self.control_points, samples_per_segment=40)
        self.center_points = self._smooth_centerline(self.center_points, iterations=2)
        self.outer_points, self.inner_points = self._compute_boundaries(self.center_points, self.base_track_width)
        self.finish_line = self.calculate_finish_line()
        self.track_polygon = self.outer_points + list(reversed(self.inner_points))
        self.curvatures = self._compute_curvatures(self.center_points)
        self.surface, self.mask = self._build_surface()
        self.dash_segments = self._build_center_dashes()

    # ----------- Track Generation Helpers -----------
    def _create_control_points(self) -> List[Point]:
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        pts = [
            (cx - 320, cy + 180), (cx - 150, cy + 190), (cx + 40, cy + 200), (cx + 260, cy + 160),
            (cx + 340, cy + 60), (cx + 360, cy - 40), (cx + 320, cy - 160), (cx + 200, cy - 230),
            (cx + 40, cy - 250), (cx - 140, cy - 240), (cx - 300, cy - 200), (cx - 370, cy - 120),
            (cx - 380, cy), (cx - 360, cy + 100)
        ]
        return [Point(x, y) for x, y in pts]

    def _catmull_rom(self, p0, p1, p2, p3, t):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t2 + (-p0 + 3*p1 - 3*p2 + p3) * t3)

    def _interpolate_centerline(self, control_points: List[Point], samples_per_segment: int = 20) -> List[Point]:
        pts = control_points[:]
        extended = [pts[-2], pts[-1]] + pts + [pts[0], pts[1]]
        result: List[Point] = []
        for i in range(2, len(extended) - 2):
            p0, p1, p2, p3 = extended[i-2], extended[i-1], extended[i], extended[i+1]
            for s in range(samples_per_segment):
                t = s / samples_per_segment
                x = self._catmull_rom(p0.x, p1.x, p2.x, p3.x, t)
                y = self._catmull_rom(p0.y, p1.y, p2.y, p3.y, t)
                result.append(Point(x, y))
        return result

    def _smooth_centerline(self, pts: List[Point], iterations: int = 1) -> List[Point]:
        if not pts:
            return pts
        n = len(pts)
        work = pts[:]
        for _ in range(iterations):
            new_pts: List[Point] = []
            for i in range(n):
                p_prev = work[(i - 1) % n]
                p = work[i]
                p_next = work[(i + 1) % n]
                nx = (p_prev.x + p.x * 2 + p_next.x) / 4.0
                ny = (p_prev.y + p.y * 2 + p_next.y) / 4.0
                new_pts.append(Point(nx, ny))
            work = new_pts
        return work

    def _compute_boundaries(self, center: List[Point], half_width: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        outer, inner = [], []
        n = len(center)
        for i in range(n):
            p_prev = center[i - 1]
            p = center[i]
            p_next = center[(i + 1) % n]
            tx = p_next.x - p_prev.x
            ty = p_next.y - p_prev.y
            length = math.hypot(tx, ty) or 1.0
            nx = -ty / length
            ny = tx / length
            outer.append((p.x + nx * half_width, p.y + ny * half_width))
            inner.append((p.x - nx * half_width, p.y - ny * half_width))
        return outer, inner

    def _compute_curvatures(self, center: List[Point]) -> List[float]:
        curv: List[float] = []
        n = len(center)
        for i in range(n):
            p_prev = center[i - 1]
            p = center[i]
            p_next = center[(i + 1) % n]
            v1x = p.x - p_prev.x; v1y = p.y - p_prev.y
            v2x = p_next.x - p.x; v2y = p_next.y - p.y
            l1 = math.hypot(v1x, v1y) or 1
            l2 = math.hypot(v2x, v2y) or 1
            v1x /= l1; v1y /= l1; v2x /= l2; v2y /= l2
            dot = max(-1.0, min(1.0, v1x * v2x + v1y * v2y))
            angle = math.acos(dot)
            curv.append(angle)
        return curv

    def _point_in_polygon(self, x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
            j = i
        return inside

    def is_on_track(self, x: float, y: float) -> bool:
        ix = int(x); iy = int(y)
        if ix < 0 or iy < 0 or ix >= SCREEN_WIDTH or iy >= SCREEN_HEIGHT:
            return False
        try:
            return self.mask.get_at((ix, iy)) == 1
        except Exception:
            return False

    def calculate_finish_line(self) -> Tuple[Point, Point]:
        if len(self.center_points) < 2:
            return Point(100, 100), Point(100, 140)
        start = self.center_points[0]
        next_point = self.center_points[5]
        dx = next_point.x - start.x
        dy = next_point.y - start.y
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_x = -dy / length * 40
            perp_y = dx / length * 40
            line_start = Point(start.x + perp_x, start.y + perp_y)
            line_end = Point(start.x - perp_x, start.y - perp_y)
            return line_start, line_end
        return Point(start.x, start.y - 40), Point(start.x, start.y + 40)

    def _build_surface(self):
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        path_points = [(p.x, p.y) for p in self.center_points]
        track_width_px = int(self.base_track_width * 2)
        if track_width_px % 2 == 1:
            track_width_px += 1
        pygame.draw.lines(surf, TRACK_GRAY, True, path_points, track_width_px)
        circle_r = int(self.base_track_width)
        for p in self.center_points:
            pygame.draw.circle(surf, TRACK_GRAY, (int(p.x), int(p.y)), circle_r)
        pygame.draw.lines(surf, WHITE, True, self.outer_points, 4)
        pygame.draw.lines(surf, WHITE, True, self.inner_points, 4)
        for i, p in enumerate(self.center_points):
            if self.curvatures[i] > 0.08 and i % 3 == 0:
                opx, opy = self.outer_points[i]
                ipx, ipy = self.inner_points[i]
                color = RED if (i // 2) % 2 == 0 else WHITE
                pygame.draw.circle(surf, color, (int(opx), int(opy)), 4)
                pygame.draw.circle(surf, color, (int(ipx), int(ipy)), 4)
        for i in range(0, len(self.center_points), 55):
            p = self.center_points[i]
            shade_val = 95 + (i * 17) % 25
            shade = (shade_val, shade_val, shade_val)
            pygame.draw.circle(surf, shade, (int(p.x), int(p.y)), 18, width=1)
        mask = pygame.mask.from_surface(surf)
        return surf, mask

    def _build_center_dashes(self):
        dashes = []
        dash_len = 25; gap = 25; acc = 0.0; draw_dash = True
        last = self.center_points[0]
        for p in self.center_points[1:]:
            seg_len = math.hypot(p.x - last.x, p.y - last.y)
            if draw_dash:
                dashes.append(((last.x, last.y), (p.x, p.y)))
            acc += seg_len
            if draw_dash and acc >= dash_len:
                acc = 0; draw_dash = False
            elif (not draw_dash) and acc >= gap:
                acc = 0; draw_dash = True
            last = p
        return dashes

    def draw(self, screen: pygame.Surface):
        if len(self.center_points) < 2:
            return
        screen.blit(self.surface, (0, 0))
        for a, b in self.dash_segments:
            pygame.draw.line(screen, YELLOW, a, b, 2)
        line_start, line_end = self.finish_line
        segments = 10
        for i in range(segments):
            t0 = i / segments; t1 = (i + 1) / segments
            sx = line_start.x + (line_end.x - line_start.x) * t0
            sy = line_start.y + (line_end.y - line_start.y) * t0
            ex = line_start.x + (line_end.x - line_start.x) * t1
            ey = line_start.y + (line_end.y - line_start.y) * t1
            color = WHITE if i % 2 == 0 else BLACK
            pygame.draw.line(screen, color, (sx, sy), (ex, ey), 8)
        dx = line_end.x - line_start.x; dy = line_end.y - line_start.y
        L = math.hypot(dx, dy) or 1; nx = dx / L; ny = dy / L
        grid_rows = 5
        for r in range(grid_rows):
            gx0 = (line_start.x + line_end.x)/2 + nx * (r * 18)
            gy0 = (line_start.y + line_end.y)/2 + ny * (r * 18)
            pygame.draw.line(screen, (200,200,200), (gx0 - ny*30, gy0 + nx*30), (gx0 + ny*30, gy0 - nx*30), 2)

class Game:
    """Główna klasa gry"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Racing Game - Gra Wyścigowa z AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        # Komponenty
        self.track = Track()
        self.cars = self.create_cars()

        # Stan
        self.generation = 1
        self.time_elapsed = 0
        self.max_time = ROUND_DURATION_SECONDS * FPS
        self.running = True

        # Wyniki
        self.best_score = 0.0
        self.best_generation = 1
        self.champion_brain = None
        self.pretrained = False

        if PRETRAIN_GENERATIONS > 0:
            self.pretrain(PRETRAIN_GENERATIONS)

    def create_cars(self) -> List[Car]:
        cars: List[Car] = []
        if len(self.track.center_points) > 10:
            start_idx = 3
            base = self.track.center_points[start_idx]
            forward = self.track.center_points[start_idx + 5]
            tx = forward.x - base.x
            ty = forward.y - base.y
            heading = math.atan2(ty, tx)
            L = math.hypot(tx, ty) or 1
            tx /= L; ty /= L
            nx = -ty; ny = tx
            rows, cols = 5, 2
            row_spacing = 30
            col_spacing = self.track.base_track_width * 0.4
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= 10:
                        break
                    lateral = (c - (cols - 1)/2) * col_spacing
                    back = r * row_spacing
                    px = base.x - tx * back + nx * lateral
                    py = base.y - ty * back + ny * lateral
                    if not self.track.is_on_track(px, py):
                        px = base.x - tx * back
                        py = base.y - ty * back
                    car = Car(px, py, heading, idx)
                    cars.append(car)
                    idx += 1
        return cars

    def update(self):
        self.time_elapsed += 1
        for car in self.cars:
            car.update(self.track)
        if self.time_elapsed >= self.max_time or not any(c.alive for c in self.cars):
            self.evolve_cars()

    def evolve_cars(self):
        self.cars.sort(key=lambda c: c.score, reverse=True)
        if self.cars and self.cars[0].score > self.best_score:
            self.best_score = self.cars[0].score
            self.best_generation = self.generation
            self.champion_brain = self.cars[0].brain.copy()
        elite = self.cars[:3]
        new_cars: List[Car] = []
        for i in range(10):
            if self.champion_brain is not None and i == 0:
                brain = self.champion_brain.copy()
            elif i < len(elite):
                brain = elite[i].brain.copy()
            else:
                parent = random.choice(elite)
                brain = parent.brain.copy()
                brain.mutate(mutation_rate=0.3, mutation_strength=0.5)
            new_cars.append(self.create_car_from_brain(brain, i))
        self.cars = new_cars
        self.generation += 1
        self.time_elapsed = 0

    def pretrain(self, generations: int):
        original = self.max_time
        self.max_time = int(8 * FPS)
        if PRETRAIN_SHOW_PROGRESS:
            self.screen.fill(BLACK)
            info = self.font.render("Pre-trening AI...", True, WHITE)
            self.screen.blit(info, (40, 40))
            pygame.display.flip()
        for g in range(generations):
            if g > 0:
                self.evolve_cars()
            self.time_elapsed = 0
            while self.time_elapsed < self.max_time and any(c.alive for c in self.cars):
                for car in self.cars:
                    car.update(self.track)
                self.time_elapsed += 1
            if PRETRAIN_SHOW_PROGRESS and (g % PRETRAIN_DRAW_INTERVAL == 0 or g == generations - 1):
                self.screen.fill(BLACK)
                t = self.font.render(f"Pre-trening: {g+1}/{generations}", True, WHITE)
                self.screen.blit(t, (40, 40))
                best = max((c.score for c in self.cars), default=0.0)
                btxt = self.small_font.render(f"Tymczasowy najlepszy: {best:.1f}", True, WHITE)
                self.screen.blit(btxt, (40, 70))
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False; return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        generations = g + 1
                        break
        if self.cars:
            self.cars.sort(key=lambda c: c.score, reverse=True)
            self.champion_brain = self.cars[0].brain.copy()
            new_cars: List[Car] = []
            for i in range(10):
                brain = self.champion_brain.copy()
                if i > 0:
                    brain.mutate(mutation_rate=0.45, mutation_strength=0.55)
                new_cars.append(self.create_car_from_brain(brain, i))
            self.cars = new_cars
        self.generation = 1
        self.time_elapsed = 0
        self.max_time = original
        self.pretrained = True

    def create_car_from_brain(self, brain: NeuralNetwork, agent_id: int) -> Car:
        if len(self.track.center_points) > 10:
            start_idx = 3
            base = self.track.center_points[start_idx]
            forward = self.track.center_points[start_idx + 5]
            tx = forward.x - base.x
            ty = forward.y - base.y
            heading = math.atan2(ty, tx)
            L = math.hypot(tx, ty) or 1
            tx /= L; ty /= L
            nx = -ty; ny = tx
            rows, cols = 5, 2
            row_spacing = 30
            col_spacing = self.track.base_track_width * 0.4
            r = agent_id // cols
            c = agent_id % cols
            lateral = (c - (cols - 1)/2) * col_spacing
            back = r * row_spacing
            px = base.x - tx * back + nx * lateral
            py = base.y - ty * back + ny * lateral
            if not self.track.is_on_track(px, py):
                px = base.x - tx * back
                py = base.y - ty * back
            car = Car(px, py, heading, agent_id)
            car.brain = brain
            return car
        car = Car(100, 100, 0, agent_id)
        car.brain = brain
        return car

    def draw_hud(self):
        hud = pygame.Rect(10, 10, 300, 200)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), hud)
        pygame.draw.rect(self.screen, WHITE, hud, 2)
        y = 20
        def line(text, font=None, dy=25):
            nonlocal y
            f = font or self.font
            surf = f.render(text, True, WHITE)
            self.screen.blit(surf, (20, y))
            y += dy
        line(f"Generacja: {self.generation}")
        remaining = max(0, (self.max_time - self.time_elapsed) // FPS)
        line(f"Czas: {remaining}s")
        line(f"Najlepszy wynik: {self.best_score:.1f}")
        line(f"(Generacja {self.best_generation})", self.small_font, 20)
        alive = sum(1 for c in self.cars if c.alive)
        line(f"Aktywni agenci: {alive}/10")
        if self.cars:
            current_best = max(c.score for c in self.cars)
            line(f"Aktualny naj.: {current_best:.1f}")
        alive_list = [c for c in self.cars if c.alive]
        if alive_list:
            lead = alive_list[0]
            line(f"Predkosc: {lead.speed:.1f}/{lead.max_speed:.1f}", self.small_font, 18)
        fps_val = self.clock.get_fps()
        line(f"FPS: {fps_val:.0f}", self.small_font, 18)

    def draw_leaderboard(self):
        board = pygame.Rect(SCREEN_WIDTH - 200, 10, 180, 300)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), board)
        pygame.draw.rect(self.screen, WHITE, board, 2)
        title = self.font.render("Ranking", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH - 180, 20))
        y = 50
        for i, car in enumerate(sorted(self.cars, key=lambda c: c.score, reverse=True)[:10]):
            status = "✓" if car.alive else "✗"
            color = car.color if car.alive else GRAY
            rtxt = self.small_font.render(f"{i+1}. Agent {car.agent_id} {status}", True, color)
            self.screen.blit(rtxt, (SCREEN_WIDTH - 190, y))
            stxt = self.small_font.render(f"   {car.score:.1f}", True, color)
            self.screen.blit(stxt, (SCREEN_WIDTH - 190, y + 12))
            y += 28

    def draw(self):
        self.screen.fill(GRASS_GREEN)
        self.track.draw(self.screen)
        for car in self.cars:
            car.draw(self.screen)
        self.draw_hud()
        self.draw_leaderboard()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.evolve_cars()
                elif event.key == pygame.K_r:
                    self.generation = 1
                    self.time_elapsed = 0
                    self.best_score = 0
                    self.best_generation = 1
                    self.cars = self.create_cars()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

def main():
    """Funkcja główna"""
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
