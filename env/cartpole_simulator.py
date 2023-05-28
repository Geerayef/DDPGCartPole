import pygame
from tkinter import filedialog, Tk
from .scenery import Scenery

resolution = (1000, 300)


def cartpole_simulator():
    Tk().withdraw()

    pygame.init()
    pygame.display.set_caption("Cart-pole simulator")
    surface = pygame.display.set_mode(resolution)
    clock = pygame.time.Clock()

    if pygame.font.match_font("Monospace", True):
        font = pygame.font.SysFont("Monospace", 20, True)
    elif pygame.font.match_font("Courier New", True):
        font = pygame.font.SysFont("Courier New", 20, True)
    else:
        font = pygame.font.Font(None, 20)

    scenery = Scenery(surface)

    run = True
    sum_dt = 0
    fps = 0
    sum_fps = 0
    frame_count = 0
    avg_fps = 0

    while run:
        frame_count += 1
        dt = clock.get_time()
        sum_dt += dt
        if dt > 0:
            fps = 1000.0 / dt
        sum_fps += fps
        if sum_dt >= 100:
            avg_fps = sum_fps / frame_count
            sum_fps = 0
            frame_count = 0
            sum_dt = 0

        scenery.tick(dt / 1000.0)

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    scenery.key_pressed("left")
                elif event.key == pygame.K_RIGHT:
                    scenery.key_pressed("right")
                elif event.key == pygame.K_RETURN:
                    scenery.reset()
                elif event.key == pygame.K_r:
                    if scenery.is_recording():
                        scenery.stop_recording()
                    else:
                        scenery.start_recording()
                elif event.key == pygame.K_p:
                    if scenery.is_playing():
                        scenery.stop_playing()
                    else:
                        scenery.start_playing(filedialog.askopenfilename())
                elif event.key == pygame.K_c:
                    filename = filedialog.askopenfilename()
                    if filename != "":
                        surface.fill((0, 0, 0))
                        msg = "Converting video ..."
                        (width, height) = font.size(msg)
                        text = font.render(msg, True, (255, 255, 255))
                        surface.blit(
                            text, ((surface.get_width() - width) / 2, (surface.get_height() - height) / 2))
                        pygame.display.update()
                        scenery.convert_recording(filename)
                elif event.key == pygame.K_ESCAPE:
                    run = False

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    scenery.key_released("left")
                elif event.key == pygame.K_RIGHT:
                    scenery.key_released("right")

            elif event.type == pygame.MOUSEMOTION:
                scenery.mouse_move(event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    scenery.mouse_down(event.pos)
                elif event.button == 4:
                    scenery.mouse_wheel(1)
                elif event.button == 5:
                    scenery.mouse_wheel(-1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    scenery.mouse_up()

            elif event.type == pygame.QUIT:
                run = False

        scenery.draw()
        text = font.render("FPS: %.1f" % avg_fps, True, (255, 255, 255))
        surface.blit(text, (5, 5))
        text_y = 5
        if pygame.time.get_ticks() % 1000 <= 500:
            msg = ""
            color = (255, 255, 255)
            if scenery.is_recording():
                msg = "RECORDING"
                color = (255, 64, 64)
            elif scenery.is_playing():
                msg = "PLAYING"
                color = (64, 255, 64)
            (width, height) = font.size(msg)
            text = font.render(msg, True, color)
            surface.blit(text, (surface.get_width() - width - 5, text_y))
            text_y += height

        pygame.display.update()
        clock.tick(50)

    pygame.quit()


if __name__ == '__main__':
    cartpole_simulator()
