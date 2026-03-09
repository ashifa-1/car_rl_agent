import pygame
import math


class Car:

    def __init__(self, x, y):

        # Position
        self.x = x
        self.y = y

        # Movement
        self.angle = 0
        self.velocity = 0

        self.max_speed = 5
        self.acceleration = 0.2
        self.turn_speed = 5

        # Car size
        self.width = 20
        self.height = 10

        # Sensor configuration
        self.ray_angles = [-90, -45, 0, 45, 90]
        self.max_ray_distance = 300


    # -----------------------------
    # Movement
    # -----------------------------

    def accelerate(self):
        self.velocity += self.acceleration
        self.velocity = min(self.velocity, self.max_speed)


    def brake(self):
        self.velocity -= self.acceleration
        self.velocity = max(self.velocity, -self.max_speed / 2)


    def turn(self, direction):
        """
        direction:
        -1 = left
         1 = right
        """
        self.angle += direction * self.turn_speed


    def update(self):

        rad = math.radians(self.angle)

        self.x += math.cos(rad) * self.velocity
        self.y += math.sin(rad) * self.velocity


    # -----------------------------
    # Drawing
    # -----------------------------

    def draw(self, surface):

        rect = pygame.Rect(0, 0, self.width, self.height)
        rect.center = (self.x, self.y)

        pygame.draw.rect(surface, (255, 0, 0), rect)


    # -----------------------------
    # Collision
    # -----------------------------

    def get_rect(self):

        rect = pygame.Rect(0, 0, self.width, self.height)
        rect.center = (self.x, self.y)

        return rect


    def check_collision(self, walls):

        car_rect = self.get_rect()

        for x1, y1, x2, y2 in walls:

            wall_rect = pygame.Rect(
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1) if abs(x2 - x1) > 1 else 2,
                abs(y2 - y1) if abs(y2 - y1) > 1 else 2
            )

            if car_rect.colliderect(wall_rect):
                return True

        return False


    # -----------------------------
    # Geometry helpers
    # -----------------------------

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def line_intersection(self, p0, p1, p2, p3):

        s1_x = p1[0] - p0[0]
        s1_y = p1[1] - p0[1]

        s2_x = p3[0] - p2[0]
        s2_y = p3[1] - p2[1]

        denom = (-s2_x * s1_y + s1_x * s2_y)

        if denom == 0:
            return None

        s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
        t = (s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom

        if 0 <= s <= 1 and 0 <= t <= 1:

            x = p0[0] + (t * s1_x)
            y = p0[1] + (t * s1_y)

            return (x, y)

        return None


    # -----------------------------
    # Ray Casting Sensors
    # -----------------------------

    def cast_rays(self, walls):

        readings = []

        for angle_offset in self.ray_angles:

            ray_angle = math.radians(self.angle + angle_offset)

            ray_end = (
                self.x + math.cos(ray_angle) * self.max_ray_distance,
                self.y + math.sin(ray_angle) * self.max_ray_distance
            )

            closest_dist = self.max_ray_distance
            closest_point = None

            for x1, y1, x2, y2 in walls:

                hit = self.line_intersection(
                    (self.x, self.y),
                    ray_end,
                    (x1, y1),
                    (x2, y2)
                )

                if hit:

                    dist = self.distance(self.x, self.y, hit[0], hit[1])

                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = hit

            readings.append((closest_dist, closest_point))

        return readings


    def draw_rays(self, surface, readings):

        for dist, point in readings:

            if point:
                pygame.draw.line(
                    surface,
                    (0, 255, 0),
                    (self.x, self.y),
                    point,
                    1
                )