"""Watch a trained agent play Flappy Bird, or play it yourself.   python play.py

A menu in the game window lists the trained agents with their final test score.
Menu: mouse or Up/Down to pick, click or Enter to start, Left/Right (or click the speed) to change the speed,
Esc to quit. In a game: space, up arrow or a click flaps when you play; Esc goes back to the menu.
"""
import importlib
import json
from pathlib import Path

import pygame

from envs.flappy import make_env

RUNS = Path("runs")
NAMES = {"q_learning/baseline": "Q-learning", "dqn/baseline": "DQN", "dqn/double": "Double DQN",
         "ppo/baseline": "PPO", "sb3_ppo/baseline": "SB3 PPO"}
SPEEDS = [1, 2, 4, 8]  # x the real 30 fps
WHITE, MUTED, ACCENT, INK = (255, 255, 255), (190, 190, 190), (237, 161, 0), (20, 20, 20)
ROW_Y, ROW_H, VISIBLE = 100, 26, 13


def entries():
    """[(label, score, run dir)] -- you first (run None), then every evaluated agent (all trained runs if none are)."""
    runs = (sorted(p.parent for p in RUNS.glob("*/*/seed*/final_eval.json"))
            or sorted(p.parent for p in RUNS.glob("*/*/seed*/summary.json")))
    order = list(NAMES)

    def key(d):
        k = d.parent.relative_to(RUNS).as_posix()
        return (order.index(k) if k in order else len(order), k, d.name)
    rows = [("You (keyboard)", "", None)]
    for d in sorted(runs, key=key):
        k = d.parent.relative_to(RUNS).as_posix()
        ev = d / "final_eval.json"
        rows.append((f"{NAMES.get(k, k)}  ·  seed {d.name[4:]}",
                     f"{json.loads(ev.read_text())['mean']:.1f}" if ev.exists() else "", d))
    return rows


def load_policy(run):
    cfg = json.loads((run / "config.json").read_text())
    return importlib.import_module(f"agents.{cfg['algo']}").load_policy(run, cfg)


def text(screen, font, words, color, **pos):
    img = font.render(words, True, color)
    return screen.blit(img, img.get_rect(**pos))


def shade(screen, alpha):
    s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    s.fill((0, 0, 0, alpha))
    screen.blit(s, (0, 0))


def menu(screen, backdrop, fonts, rows, sel, speed):
    """Shows the menu until something is chosen. Returns (row index or None to quit, speed index)."""
    big, font, small = fonts
    w = screen.get_width()
    while True:
        first = min(max(0, sel - VISIBLE + 1), max(0, len(rows) - VISIBLE))  # ponytail: scrolls one list, no paging
        screen.blit(backdrop, (0, 0))
        shade(screen, 205)
        text(screen, big, "Flappy Bird RL", WHITE, midtop=(w // 2, 18))
        text(screen, small, "choose who plays", MUTED, midtop=(w // 2, 56))
        text(screen, small, "agent", MUTED, topleft=(20, ROW_Y - 22))
        text(screen, small, "avg pipes", MUTED, topright=(w - 20, ROW_Y - 22))
        hits = {}
        for i in range(first, min(len(rows), first + VISIBLE)):
            r = pygame.Rect(10, ROW_Y + (i - first) * ROW_H, w - 20, ROW_H - 3)
            hits[i] = r
            on = i == sel
            if on:
                pygame.draw.rect(screen, ACCENT, r, border_radius=6)
            label, score, _ = rows[i]
            text(screen, font, label, INK if on else WHITE, midleft=(r.x + 10, r.centery))
            text(screen, font, score, INK if on else MUTED, midright=(r.right - 10, r.centery))
        speed_hit = text(screen, font, f"speed   <  {SPEEDS[speed]}x  >", WHITE, midtop=(w // 2, ROW_Y + VISIBLE * ROW_H + 10))
        text(screen, small, "Enter or click: play   ·   Esc: quit", MUTED, midbottom=(w // 2, screen.get_height() - 10))
        pygame.display.flip()

        e = pygame.event.wait()
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            return None, speed
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                return sel, speed
            sel = (sel + {pygame.K_UP: -1, pygame.K_DOWN: 1}.get(e.key, 0)) % len(rows)
            speed = min(max(speed + {pygame.K_LEFT: -1, pygame.K_RIGHT: 1}.get(e.key, 0), 0), len(SPEEDS) - 1)
        elif e.type == pygame.MOUSEMOTION:
            sel = next((i for i, r in hits.items() if r.collidepoint(e.pos)), sel)
        elif e.type == pygame.MOUSEWHEEL:
            sel = (sel - e.y) % len(rows)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if speed_hit.collidepoint(e.pos):
                speed = (speed + 1) % len(SPEEDS)
            elif (i := next((i for i, r in hits.items() if r.collidepoint(e.pos)), None)) is not None:
                return i, speed


def play(env, policy, fps):
    """One game. Returns its score, or None when Esc sends the player back to the menu."""
    env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": fps}
    obs, _ = env.reset()
    while True:
        flap = 0
        for e in pygame.event.get():  # also keeps the window responsive while an agent plays
            if e.type == pygame.QUIT:
                raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return None
            if (e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_UP)) or e.type == pygame.MOUSEBUTTONDOWN:
                flap = 1
        obs, _, term, trunc, info = env.step(flap if policy is None else policy(obs))
        if term or trunc:
            return info["score"]


def game_over(screen, fonts, score):
    """Score banner over the last frame. True = play again, False = back to the menu."""
    big, font, small = fonts
    w, h = screen.get_size()
    shade(screen, 150)
    text(screen, big, f"{score} pipes", WHITE, center=(w // 2, h // 2 - 20))
    text(screen, small, "Enter or click: again   ·   Esc: menu", WHITE, center=(w // 2, h // 2 + 18))
    pygame.display.flip()
    pygame.time.wait(400)  # a flap pressed at the moment of the crash must not restart the game
    pygame.event.clear()
    while True:
        e = pygame.event.wait()
        if e.type == pygame.QUIT:
            raise SystemExit
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            return False
        if (e.type == pygame.KEYDOWN and e.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)) or \
                (e.type == pygame.MOUSEBUTTONDOWN and e.button == 1):
            return True


def main():
    pygame.init()
    rows = entries()
    env = make_env(max_steps=None, render_mode="human")
    env.reset()  # opens the window and draws a game frame, reused as the menu's background
    pygame.display.set_caption("Flappy Bird RL")
    screen = pygame.display.get_surface()
    backdrop = screen.copy()
    backdrop.fill(backdrop.get_at((8, 60)), (screen.get_width() // 2 - 40, 40, 80, 60))  # hide the game's score digit
    fonts = [pygame.font.SysFont("segoeui,helvetica,arial", s, bold=b) for s, b in ((30, True), (16, False), (13, False))]
    policies, sel, speed = {}, 0, 0
    try:
        while True:
            choice, speed = menu(screen, backdrop, fonts, rows, sel, speed)
            if choice is None:
                break
            sel, run = choice, rows[choice][2]
            if run is not None and run not in policies:
                text(screen, fonts[1], "loading...", WHITE, midbottom=(screen.get_width() // 2, screen.get_height() - 34))
                pygame.display.flip()
                policies[run] = load_policy(run)
            while (score := play(env, policies.get(run), 30 * SPEEDS[speed])) is not None and game_over(screen, fonts, score):
                pass
    except SystemExit:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
