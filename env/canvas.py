import pygame
import pygame.gfxdraw
import math


class Canvas:
    surface = None
    ppm = 150
    x = 0
    y = 0
    offset_x = 0
    offset_y = 0

    def __init__(self, surface):
        self.surface = surface
        # self.offset_x = 4 * float(surface.get_width()) / 5
        self.offset_x = float(surface.get_width()) / 2
        self.offset_y = float(surface.get_height()) / 2

    def __copy__(self):
        canvas = Canvas(self.surface.copy())
        canvas.ppm = self.ppm
        canvas.x = self.x
        canvas.y = self.y
        canvas.offset_x = self.offset_x
        canvas.offset_y = self.offset_y
        return canvas

    def get_size(self):
        return self.surface.get_size()

    def set_vertical_offset_at(self, surface_ratio):
        self.offset_y = float(self.surface.get_height()) * surface_ratio

    def zoom(self, factor):
        self.ppm *= factor

    def move_focus_by(self, position):
        (x, y) = position
        self.x += x
        self.y += y

    def to_screen(self, position):
        (x, y) = position
        sx = (x - self.x) * self.ppm + self.offset_x
        sy = self.offset_y - (y - self.y) * self.ppm
        sx = int(round(sx))
        sy = int(round(sy))
        return sx, sy

    def to_canvas(self, position):
        (x, y) = position
        cx = (float(x) - self.offset_x) / self.ppm + self.x
        cy = (self.offset_y - float(y)) / self.ppm + self.y
        return cx, cy

    def rotate(self, position, anchor, angle):
        (x, y) = position
        (anchor_x, anchor_y) = anchor
        rad = math.radians(angle)
        sin = math.sin(rad)
        cos = math.cos(rad)
        mx = x - anchor_x
        my = y - anchor_y
        rx = mx * cos - my * sin
        ry = mx * sin + my * cos
        return rx + anchor_x, ry + anchor_y

    def draw_rectangle(self, position_from, position_to, color, anchor=(0, 0), rotation=0):
        (x0, y0) = position_from
        (x1, y1) = position_to
        if rotation == 0:
            sx0, sy0 = self.to_screen((x0, y0))
            sx1, sy1 = self.to_screen((x1, y1))
            points = [(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)]
        else:
            rx0, ry0 = self.rotate((x0, y0), anchor, rotation)
            rx1, ry1 = self.rotate((x1, y0), anchor, rotation)
            rx2, ry2 = self.rotate((x1, y1), anchor, rotation)
            rx3, ry3 = self.rotate((x0, y1), anchor, rotation)
            sx0, sy0 = self.to_screen((rx0, ry0))
            sx1, sy1 = self.to_screen((rx1, ry1))
            sx2, sy2 = self.to_screen((rx2, ry2))
            sx3, sy3 = self.to_screen((rx3, ry3))
            points = [(sx0, sy0), (sx1, sy1), (sx2, sy2), (sx3, sy3)]
        pygame.gfxdraw.aapolygon(self.surface, points, color)
        pygame.gfxdraw.filled_polygon(self.surface, points, color)

    def draw_circle(self, position, radius, color, anchor=(0, 0), rotation=0):
        if rotation == 0:
            sx, sy = self.to_screen(position)
        else:
            rx, ry = self.rotate(position, anchor, rotation)
            sx, sy = self.to_screen((rx, ry))
        sx = int(sx)
        sy = int(sy)
        sr = int(round(radius * self.ppm))
        pygame.gfxdraw.aacircle(self.surface, sx, sy, sr, color)
        pygame.gfxdraw.filled_circle(self.surface, sx, sy, sr, color)

    def draw_arrow(self, position_from, position_to, color, width):
        (x0, y0) = position_from
        (x1, y1) = position_to
        dx = x1 - x0
        dy = y1 - y0
        angle = math.degrees(math.atan2(dy, dx))
        length = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        head = 2.0 * width
        p0x = x0
        p0y = y0 - width / 2.0
        p1x = x0 + length - head
        p1y = y0 + width / 2.0
        self.draw_rectangle((p0x, p0y), (p1x, p1y), color, (x0, y0), angle)

        px, py = self.rotate((p1x, y0), (x0, y0), angle)
        p0x, p0y = self.rotate((px + head, py), (px, py), angle - 90)
        p1x, p1y = (x1, y1)
        p2x, p2y = self.rotate((px + head, py), (px, py), angle + 90)
        points = [
            self.to_screen((p0x, p0y)),
            self.to_screen((p1x, p1y)),
            self.to_screen((p2x, p2y))
        ]
        pygame.gfxdraw.aapolygon(self.surface, points, color)
        pygame.gfxdraw.filled_polygon(self.surface, points, color)
