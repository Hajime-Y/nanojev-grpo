"""Replay recorded Predict Position episodes side by side and write an animated GIF.

    python scripts/pp_video.py --seed 9300761 --output docs/assets/pp_train.gif \
        --panel "Before training=runs_pp/final/base_test_greedy.jsonl" \
        --panel "SFT=runs_pp/final/sft_test_greedy.jsonl" \
        --panel "GRPO=runs_pp/final/grpo_test_greedy.jsonl"

Each panel is one episode of an output file of `pp_rl.py eval` or `pp_probe.py play`. No model runs here: the
recorded action list is fed back to the same environment with the same seed, one frame is captured per game
tick, and the replayed outcome must equal the recorded one. Under each panel a bar shows the label, the action
chosen at that moment and, once the episode has ended, its result. The yellow marker above the monster is
drawn by this script for the viewer; it is not part of the game.

Needs ViZDoom and Pillow (CPU only).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_doom_env import UnifiedDoomEnv  # noqa: E402

CROP_TOP, PANEL_W, PANEL_H, BAR_H, GAP = 44, 320, 196, 44, 6
ACTIONS = {"left": "turn left", "right": "turn right", "shoot": "shoot", "noop": "wait", None: "start"}
INK, DIM, BAR, CHIP = (236, 233, 228), (150, 146, 140), (24, 23, 22), (58, 56, 53)
FIRE, HIT, MISS, MARK = (214, 72, 48), (74, 160, 96), (190, 60, 52), (255, 214, 64)


class Recorder:
    """Forwards everything to the real DoomGame and keeps the screen and the monster box after every tick."""

    def __init__(self, game):
        self._game_, self.frames, self.boxes = game, [], []

    def __getattr__(self, name):
        return getattr(self._game_, name)

    def grab(self):
        import numpy as np
        state = self._game_.get_state()
        if state is not None and state.screen_buffer is not None:
            self.frames.append(np.array(state.screen_buffer, copy=True))
            self.boxes.append(next(((int(l.x), int(l.y), int(l.width), int(l.height)) for l in state.labels
                                    if l.object_name != "DoomPlayer" and l.width > 0 and l.height > 0), None))
        elif self.frames:
            self.frames.append(self.frames[-1])
            self.boxes.append(None)

    def make_action(self, *args):
        reward = self._game_.make_action(*args)
        self.grab()
        return reward


def ammo(observation):
    return json.loads(observation["state"])["observed_history"][-1]["ammo"]


def replay(row):
    """Returns the frames of one recorded episode as (screen, (decision index, action), monster box)."""
    env = UnifiedDoomEnv(row["case"]["spec"])
    env._initialize()
    recorder = Recorder(env._game)
    env._game = recorder
    try:
        observation, info = env.reset(row["case"]["seed"])
        recorder.frames.clear()
        recorder.boxes.clear()
        recorder.grab()
        marks, done, index = [(0, None)], False, 0
        actions = [decision["action"] for decision in row["decisions"]]
        while not done:
            if index >= len(actions) and ammo(observation) > 0:
                raise RuntimeError("recorded actions ran out while the rocket was still available")
            # After the rocket is gone the evaluation driver only waits; do the same here.
            action = actions[index] if index < len(actions) and ammo(observation) > 0 else "noop"
            before = len(recorder.frames)
            observation, _, done, _, info = env.step(action)
            marks += [(index, action)] * (len(recorder.frames) - before)
            index += 1
        if info["success"] != row["success"] or info["terminal_reason"] != row["outcome"]:
            raise RuntimeError(f"replay of {row['key']} ended differently from the record")
        return list(zip(recorder.frames, marks, recorder.boxes))
    finally:
        env._game = recorder._game_
        env.close()


def load_episode(path, seed):
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            if row["case"]["seed"] == seed:
                return row
    raise SystemExit(f"seed {seed} is not in {path}")


def result_text(row):
    if row["success"]:
        return "hit", HIT
    return ("miss", MISS) if row["fired"] else ("timed out, never fired", MISS)


def draw_panel(label, row, frames, tick, fonts):
    from PIL import Image, ImageDraw
    name_font, chip_font = fonts
    image = Image.new("RGB", (PANEL_W, PANEL_H + BAR_H), BAR)
    last = len(frames) - 1
    done = tick >= last
    frame, (decision, action), box = frames[min(tick, last)]
    image.paste(Image.fromarray(frame).crop((0, CROP_TOP, 320, CROP_TOP + PANEL_H)), (0, 0))
    draw = ImageDraw.Draw(image)
    if box and not done:
        cx, top = box[0] + box[2] // 2, box[1] - CROP_TOP - 5
        if 8 <= top < PANEL_H:
            draw.polygon([(cx - 5, top - 7), (cx + 5, top - 7), (cx, top)], fill=MARK)
    draw.rectangle((0, PANEL_H, PANEL_W, PANEL_H + 3), fill=CHIP)  # elapsed game time out of 300 ticks
    draw.rectangle((0, PANEL_H, int(PANEL_W * min(tick, last) / 300), PANEL_H + 3), fill=DIM)
    y = PANEL_H + 12
    draw.text((10, y + 12), label, font=name_font, fill=INK, anchor="lm")
    if done:
        text, fill, ink = (*result_text(row), INK)
    elif row["fired"] and row["fire_decision"] is not None and decision is not None and decision > row["fire_decision"]:
        text, fill, ink = "fired", CHIP, DIM
    else:
        text, fill, ink = ACTIONS[action], (FIRE if action == "shoot" else CHIP), INK
    width = draw.textlength(text, font=chip_font) + 16
    draw.rounded_rectangle((PANEL_W - 8 - width, y, PANEL_W - 8, y + 24), 4, fill=fill)
    draw.text((PANEL_W - width, y + 12), text, font=chip_font, fill=ink, anchor="lm")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--panel", action="append", required=True, metavar="LABEL=RECORDS.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tick-step", type=int, default=2, help="game ticks per GIF frame")
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("--hold-ms", type=int, default=1800, help="how long the last frame stays")
    parser.add_argument("--font", help="TrueType font file for the labels (default: Pillow's built-in font)")
    args = parser.parse_args()
    from PIL import Image, ImageFont

    def font(size):
        return ImageFont.truetype(args.font, size) if args.font else ImageFont.load_default(size=size)

    fonts = (font(15), font(14))
    panels = []
    for spec in args.panel:
        label, path = spec.split("=", 1)
        row = load_episode(path, args.seed)
        panels.append((label, row, replay(row)))
    longest = max(len(frames) for _, _, frames in panels)
    width = PANEL_W * len(panels) + GAP * (len(panels) - 1)
    canvases = []
    for tick in list(range(0, longest, args.tick_step)) + [longest - 1]:
        canvas = Image.new("RGB", (width, PANEL_H + BAR_H), (0, 0, 0))
        for i, (label, row, frames) in enumerate(panels):
            canvas.paste(draw_panel(label, row, frames, tick, fonts), (i * (PANEL_W + GAP), 0))
        canvases.append(canvas)
    # One palette for the whole clip (taken from evenly spaced frames), so that only the pixels that change
    # between frames are stored.
    samples = canvases[::max(1, len(canvases) // 12)] + [canvases[-1]]
    sheet = Image.new("RGB", (width, (PANEL_H + BAR_H) * len(samples) + 80))
    for i, sample in enumerate(samples):
        sheet.paste(sample, (0, i * (PANEL_H + BAR_H)))
    for i, color in enumerate((MARK, HIT, MISS, FIRE, INK, DIM, BAR, CHIP)):  # keep the overlay colours exact
        sheet.paste(color, (i * width // 8, sheet.height - 80, (i + 1) * width // 8, sheet.height))
    palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    images = [canvas.quantize(palette=palette, dither=Image.Dither.NONE) for canvas in canvases]
    durations = [args.frame_ms] * len(images)
    durations[-1] = args.hold_ms
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(json.dumps({"output": args.output, "frames": len(images), "size": [width, PANEL_H + BAR_H],
                      "episodes": [{"label": label, "ticks": len(frames) - 1, "success": row["success"],
                                    "outcome": row["outcome"]} for label, row, frames in panels]}))


if __name__ == "__main__":
    main()
