import math
from util.flags import TRACE


class Cart:
    width = 1.0
    cart_weight = 10.0
    pole_weight = 1.0
    pole_length = 1.0
    # Cart
    position = (0.0, 0.0)
    speed = 0.0
    acceleration = 0.0
    # Pole
    theta = 0.0
    theta_speed = 0.0
    theta_acceleration = 0.0

    terminated = False
    # Pole angle threshold for termination
    # - from observer's perspective: 20 degrees
    theta_threshold = 70
    damping = 0
    theta_damping = 0
    actions_per_second = 20
    _time_since_action = 0.0
    _last_action = 0.0

    def __init__(self):
        pass

    def get_current_state(self):
        return self.position[0], self.speed, self.theta, self.theta_speed

    def stop(self):
        self.speed = 0.0
        self.acceleration = 0.0

    def straighten(self):
        self.theta = 0.0
        self.theta_speed = 0.0
        self.theta_acceleration = 0.0

    def move_by(self, position):
        (x, y) = position
        x0, y0 = self.position
        self.position = (x0 + x, y0 + y)

    def reset(self):
        self.stop()
        self.straighten()
        self.position = (0, 0)

    def get_bumpers(self):
        (cx, cy) = self.position
        wheel = self.width / 10
        x0 = cx - self.width / 2
        y0 = cy + wheel
        x1 = x0 + self.width
        y1 = self.width / 6 + wheel
        y = (y0 + y1) / 2.0
        return [(x0, y), (x1, y)]

    def collides_with(self, position):
        (x, y) = position
        (cx, cy) = self.position
        wheel = self.width / 10
        x0 = cx - self.width / 2
        y0 = cy + wheel
        x1 = x0 + self.width
        y1 = self.width / 6 + wheel
        return x0 <= x <= x1 and y0 <= y <= y1

    def tick(self, f, g, dt):
        self._time_since_action += dt
        (x, y) = self.position

        rad = math.radians(self.theta)
        sin = math.sin(rad)
        cos = math.cos(rad)
        theta_speed_rad = math.radians(self.theta_speed)

        a = self.pole_weight * self.pole_length * math.pow(theta_speed_rad, 2) * sin
        b = self.pole_weight * g * sin * cos
        c = self.cart_weight + self.pole_weight - self.pole_weight * cos * cos
        self.acceleration = (f - a + b) / c - self.damping * self.speed
        self.speed += self.acceleration * dt
        x += self.speed * dt
        self.position = (x, y)

        if TRACE:
            print(f"~~~~~ DT {dt}")
            print(f"~~~~~ New Cart Position: {self.position}")

        a = self.acceleration / self.pole_length * cos
        b = g / self.pole_length * sin
        self.theta_acceleration = (
            math.degrees(a + b) - self.theta_damping * self.theta_speed
        )
        self.theta_speed += self.theta_acceleration * dt
        self.theta += self.theta_speed * dt
        if self.theta < 180:
            self.theta += 360
        if self.theta >= 180:
            self.theta -= 360

        if TRACE:
            print(f"~~~~~ Theta angle: {self.theta}")
        self.terminated = bool(
                self.position[0] > 5.
                or self.position[0] < -5.
                or self.theta > self.theta_threshold
                or self.theta < -self.theta_threshold
        )

    def draw(self, canvas):
        (x, y) = self.position
        wheel = self.width / 10
        pole = self.width / 20
        x0 = x - pole / 2.0
        y0 = y + wheel + self.width / 6
        x1 = x0 + pole
        y1 = y0 + self.pole_length
        canvas.draw_rectangle(
                (x0, y0), (x1, y1),
                (64, 32, 32), (x, y0), self.theta
                )
        canvas.draw_circle((x, y0 - 0.5 * pole), 1.2 * pole, (128, 0, 0))
        canvas.draw_circle((x, y1), 1.5 * pole, (0, 0, 0), (x, y0), self.theta)

        x0 = x - self.width / 2
        y0 = y + wheel
        x1 = x0 + self.width
        y1 = y0 + self.width / 6
        canvas.draw_rectangle((x0, y0), (x1, y1), (128, 0, 0))
        canvas.draw_circle((x0 + 2.0 * wheel, y0), wheel, (0, 0, 0))
        canvas.draw_circle((x0 + 2.0 * wheel, y0), wheel / 2.0, (64, 32, 32))
        canvas.draw_circle((x1 - 2.0 * wheel, y0), wheel, (0, 0, 0))
        canvas.draw_circle((x1 - 2.0 * wheel, y0), wheel / 2.0, (64, 32, 32))
