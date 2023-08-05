import os
import struct
import copy
import glob
import subprocess
import pygame
import numpy as np
import math
from .canvas import Canvas
from .cart import Cart
from util.flags import TRACE


class Scenery:
    gravity = 9.81

    _canvas = None
    _cart = None
    _manual_force = 100.0
    _action = 0.0
    _grab = ""
    _pos = (0, 0)
    _automatic = False
    _playing = False
    _recording = False
    _frozen = False
    _data = ""
    _p_data = 0
    _start_time = 0
    _max_ang_velo = 0.1

    def __init__(self, max_steps, surface):
        self._canvas = Canvas(surface)
        self._canvas.set_vertical_offset_at(7.0 / 8.0)
        self._cart = Cart(max_steps)
        self._max_steps = max_steps

    def reset(self):
        self._action = 0
        self._cart.reset()
        self._cart.position = (0.0, 0.0)

    def get_current_state(self):
        return self._cart.get_current_state()

    def sigmoid(self, x):
        return ( 1.0 / (1.0 + math.exp(-x)) )

    def get_reward(self, state, terminated, steps, action):
        position, velocity, angle, angular_velocity = state

        # a = abs(angular_velocity)
        # self._max_ang_velo = a if a > self._max_ang_velo else self._max_ang_velo
        # ang_velo_norm = abs(angular_velocity) / self._max_ang_velo
        # swing_speed = self.sigmoid(ang_velo_norm)

        # time = steps / self._max_steps
        upright = 1 - abs(angle) / self._cart.theta_threshold
        # upright = np.cos(np.radians(abs(angle)))
        centred = 1 - abs(position) / self._cart.position_range

        reward = ( 0.5 * upright ) + ( 0.5 * centred )
        return -1 if terminated else reward


    def switch_automatic(self):
        self._automatic = not self._automatic
        if self._automatic:
            self._experiment = False
            result = self._cart.planner_start()
            if result != "running" and result != "OK":
                print("Failed to activate the planner: " + result)
        else:
            self._cart.planner_stop()

    def key_pressed(self, key):
        if self._playing:
            return
        self._frozen = False
        if key == "right":
            self._apply_action(1)
        elif key == "left":
            self._apply_action(-1)

    def key_released(self, key):
        if self._playing:
            return
        if key == "right":
            self._apply_action(-1)
        elif key == "left":
            self._apply_action(1)

    def mouse_move(self, position):
        (x, y) = position
        if self._grab == "world":
            (x0, y0) = self._pos
            (x1, y1) = self._canvas.to_canvas((x, y))
            self._canvas.move_focus_by((x0 - x1, y0 - y1))
            self._pos = self._canvas.to_canvas((x, y))
        elif self._grab == "cart":
            (x0, y0) = self._pos
            (x1, y1) = self._canvas.to_canvas((x, y))
            self._cart.move_by((x1 - x0, 0))
            self._pos = self._canvas.to_canvas((x, y))

    def mouse_down(self, position):
        (x, y) = position
        self._pos = self._canvas.to_canvas((x, y))
        if self._cart.collides_with(self._pos):
            self._action = 0
            self._cart.stop()
            self._cart.straighten()
            self._grab = "cart"
        else:
            self._grab = "world"

    def mouse_up(self):
        self._grab = "nothing"

    def mouse_wheel(self, direction):
        if direction > 0:
            self._canvas.zoom(1.1)
        elif direction < 0:
            self._canvas.zoom(0.9)

    def _apply_action(self, direction):
        self._action += direction
        if self._action > 1:
            self._action = 1
        if self._action < -1:
            self._action = -1

    def tick(self, time, steps):
        if self._frozen:
            self._frozen = False

        if not self._frozen:
            self._cart.tick(
                self._action * self._manual_force,
                self.gravity,
                time,
                steps
            )
        if self._recording:
            self.save_frame()
        elif self._playing:
            self.load_frame()

    def post_tick(self, steps, action):
        current_state = self.get_current_state()
        reward = self.get_reward(current_state, self._cart.terminated, steps, action)
        return current_state, reward, self._cart.terminated

    def draw(self, canvas=None):
        active_canvas = canvas
        if active_canvas is None:
            active_canvas = self._canvas

        active_canvas.surface.fill((64, 128, 128))
        x0, y0 = active_canvas.to_canvas((0, 0))
        x1, y1 = active_canvas.to_canvas(active_canvas.get_size())
        active_canvas.draw_rectangle((-0.02, 0), (0.02, y0), (192, 192, 192))
        active_canvas.draw_rectangle((x0, 0), (x1, y1), (32, 32, 32))
        self._cart.draw(active_canvas)

        if self._action != 0:
            width = self._cart.width / 5.0
            bumpers = self._cart.get_bumpers()
            if self._action > 0:
                (x1, y1) = bumpers[0]
                (x0, y0) = (x1 - width, y1)
            elif self._action < 0:
                (x1, y1) = bumpers[1]
                (x0, y0) = (x1 + width, y1)
            active_canvas.draw_arrow((x0, y0), (x1, y1), (0, 0, 128), 0.02)

    def start_recording(self):
        self.stop_playing()
        self.stop_recording()
        self._data = bytes("", "utf-8")
        self._data += struct.pack("4s", b"cart")
        self._data += struct.pack("f", float(self._canvas.ppm))
        self._data += struct.pack("f", float(self._canvas.x))
        self._data += struct.pack("f", float(self._canvas.y))
        self._data += struct.pack("f", float(self._canvas.offset_x))
        self._data += struct.pack("f", float(self._canvas.offset_y))
        self._data += struct.pack("f", float(self._cart.width))
        self._data += struct.pack("f", float(self._cart.pole_length))
        self._start_time = pygame.time.get_ticks()
        self._recording = True
        self.save_frame()

    def save_frame(self):
        time = float(pygame.time.get_ticks() - self._start_time) / 1000.0
        (x, y) = self._cart.position
        self._data += struct.pack("f", float(time))
        self._data += struct.pack("f", float(x))
        self._data += struct.pack("f", float(self._cart.theta))

    def stop_recording(self):
        if self._recording:
            filename = "recording1.rec"
            count = 1
            while os.path.isfile(filename):
                count += 1
                filename = "recording" + str(count) + ".rec"
            f = open(filename, "wb")
            f.write(self._data)
            f.close()
            self._recording = False

    def read_float(self):
        if self._p_data + 4 >= len(self._data):
            return float(0), True
        data = self._data[self._p_data:self._p_data + 4]
        self._p_data += 4
        (value,) = struct.unpack("f", data)
        return value, False

    def start_playing(self, filename):
        self.stop_playing()
        self.stop_recording()
        if not os.path.isfile(filename):
            return
        f = open(filename, "rb")
        self._data = f.read()
        f.close()
        if self._data[0:4] != bytes("cart", "utf-8"):
            return
        self._p_data = 4
        ppm, eof = self.read_float()
        focus_x, eof = self.read_float()
        focus_y, eof = self.read_float()
        offset_x, eof = self.read_float()
        offset_y, eof = self.read_float()
        cart_width, eof = self.read_float()
        pole_length, eof = self.read_float()
        if not eof:
            self._canvas.ppm = ppm
            self._canvas.x = focus_x
            self._canvas.y = focus_y
            self._canvas.offset_x = offset_x
            self._canvas.offset_y = offset_y
            self._cart.width = cart_width
            self._cart.pole_length = pole_length
            self._start_time = pygame.time.get_ticks()
            self._playing = True
            self._frozen = True
            self.load_frame()

    def load_frame(self, current_time=None):
        if current_time is None:
            current_time = float(pygame.time.get_ticks() -
                                 self._start_time) / 1000.0
        frame_valid = False
        while self._playing and not frame_valid:
            time, eof = self.read_float()
            x, eof = self.read_float()
            theta, eof = self.read_float()
            if not eof:
                if time >= current_time:
                    frame_valid = True
            else:
                self._playing = False
        if frame_valid:
            self._cart.position = (x, 0)
            self._cart.theta = theta

    def stop_playing(self):
        if self._playing:
            self._playing = False
            self._file.close()

    def is_recording(self):
        return self._recording

    def is_playing(self):
        return self._playing

    def convert_frame(self, time):
        current_time = float(time - self._start_time) / 1000.0
        frame_valid = False
        while self._playing and not frame_valid:
            time, eof = self.read_float()
            x, eof = self.read_float()
            theta, eof = self.read_float()
            if not eof:
                if time >= current_time:
                    frame_valid = True
            else:
                self._playing = False
        if frame_valid:
            self._cart.position = (x, 0)
            self._cart.theta = theta

    def convert_recording(self, filename):
        if not os.path.isfile(filename):
            return

        try:
            subprocess.check_output("ffmpeg -version")
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"Cannot run ffmpeg! {e}")
            return

        f = open(filename, "rb")
        self._data = f.read()
        f.close()
        if self._data[0:4] != bytes("cart", "utf-8"):
            return
        self._p_data = 4

        canvas = copy.copy(self._canvas)
        original_cart_width = self._cart.width
        original_pole_length = self._cart.pole_length
        original_cart_position = self._cart.position
        original_cart_theta = self._cart.theta

        ppm, eof = self.read_float()
        focus_x, eof = self.read_float()
        focus_y, eof = self.read_float()
        offset_x, eof = self.read_float()
        offset_y, eof = self.read_float()
        cart_width, eof = self.read_float()
        pole_length, eof = self.read_float()
        if not eof:
            canvas.ppm = ppm
            canvas.x = focus_x
            canvas.y = focus_y
            canvas.offset_x = offset_x
            canvas.offset_y = offset_y
            self._cart.width = cart_width
            self._cart.pole_length = pole_length
            self._start_time = pygame.time.get_ticks()
            self._playing = True
            self._frozen = True
        else:
            print("Unrecognized recording format!")
            return

        fps = 50
        count = 0
        while self._playing:
            self.load_frame(float(count) / float(fps))
            self.draw(canvas)
            pygame.image.save(canvas.surface, "_" +
                              str(count + 1).zfill(6) + ".png")
            count += 1

        (x, y) = canvas.get_size()
        args = " -r " + str(fps)
        args += " -s " + str(x) + "x" + str(y)
        args += " -i " + "_%06d.png"
        args += " -vcodec libx264 -crf 25 "
        args += filename.split("/")[-1].split(".")[0] + ".mp4"
        subprocess.check_output("ffmpeg" + args)

        for png_file in glob.glob("./_*.png"):
            os.remove(png_file)

        self._cart.width = original_cart_width
        self._cart.pole_length = original_pole_length
        self._cart.position = original_cart_position
        self._cart.theta = original_cart_theta
