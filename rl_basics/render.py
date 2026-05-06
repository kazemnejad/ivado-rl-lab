"""Tile-based pixel-art renderer in Farama CliffWalking style.

Public API used by ``rl_basics.viz``::

    build_static_bg(walls, goal_xy, start_xy)              -> RGBA PIL.Image
    composite_trail(canvas, states, current_i)             -> RGBA PIL.Image
    composite_elf(canvas, state, elf_key, bump=False)      -> RGBA PIL.Image
    infer_elf_keys(states, start_facing="elf_down")        -> list[str]
    render_static(walls, goal_xy, start_xy, out_path)      -> None
    render_traj_gif(walls, goal_xy, start_xy, states,
                    bump_idx, out_path, title="", fps=6)   -> None

Sprite assets ship with the package at ``rl_basics/_assets/farama/``.

Convention (mirrors gymnasium/envs/toy_text/cliffwalking.py):
  * mountain_bg1 / mountain_bg2 alternating bg via (y%2)^(x%2) checkerboard
  * mountain_cliff (4 rotations) for walls; brown rocky edge on every floor-facing side
  * stool for start, cookie for goal
  * elf_{up,down,left,right} for agent (last-action / inferred-from-delta direction)

Wall-bump frames get a yellow/orange burst overlay on the agent's cell. A
"bump" is a step where the agent tried to move but didn't (blocked by a wall
or grid-edge clamp). Free-movement slips (random direction in open space)
are intentionally NOT highlighted — they create too much visual noise.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw

# Sprite assets live next to this module so a pip-installed wheel ships them.
IMG_DIR = Path(__file__).resolve().parent / "_assets" / "farama"

# Native cell size from Farama (40x40 for tiles, 32x32 elf scaled to 40 below).
CELL = 64  # render at higher res for crisp 1080p-ish output, NEAREST upscaling


def _load_tile(name: str) -> Image.Image:
    """Load a 40x40 cliff/bg tile and upscale to CELL with NEAREST (pixel-art preserving)."""
    return Image.open(IMG_DIR / name).convert("RGBA").resize(
        (CELL, CELL), Image.NEAREST
    )


def _load_entity_fitted(
    name: str,
    *,
    height_scale: float | None = None,
    max_dim_scale: float | None = None,
) -> Image.Image:
    """Load entity sprite, crop to its non-transparent bbox so empty padding
    doesn't shrink the figure, then scale (NEAREST, aspect-preserving):
      - height_scale=k:  target HEIGHT = k * CELL
      - max_dim_scale=k: target MAX(W, H) = k * CELL
    Native sprites have ~25-30% transparent margin so cropping first gives
    a visibly larger figure for the same logical scale."""
    img = Image.open(IMG_DIR / name).convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    cw, ch = img.size
    if height_scale is not None:
        target_h = int(CELL * height_scale)
        target_w = int(round(cw * target_h / ch))
    elif max_dim_scale is not None:
        target_max = int(CELL * max_dim_scale)
        if cw >= ch:
            target_w = target_max
            target_h = int(round(ch * target_max / cw))
        else:
            target_h = target_max
            target_w = int(round(cw * target_max / ch))
    else:
        target_w, target_h = cw, ch
    return img.resize((target_w, target_h), Image.NEAREST)


# Lazy-loaded so importing this module doesn't fail when assets are missing
# at install time (e.g. running tests from a fresh clone before sdist build).
_TILES: dict[str, Image.Image] | None = None
_CLIFF_ROT: dict[str, Image.Image] | None = None
_DARK_VOID_TILE: Image.Image | None = None
_ENT_TILES: dict[str, Image.Image] | None = None


def _ensure_loaded() -> None:
    global _TILES, _CLIFF_ROT, _DARK_VOID_TILE, _ENT_TILES
    if _TILES is not None:
        return
    _TILES = {
        "bg1": _load_tile("mountain_bg1.png"),
        "bg2": _load_tile("mountain_bg2.png"),
    }
    base = _load_tile("mountain_cliff.png")
    _CLIFF_ROT = {
        "t": base,                # 0°:   brown edge on TOP
        "l": base.rotate(90),     # 90°:  top -> left,   brown on LEFT
        "b": base.rotate(180),    # 180°: top -> bottom, brown on BOTTOM
        "r": base.rotate(270),    # 270°: top -> right,  brown on RIGHT
    }
    # Pure dark-body tile: per-pixel min of all 4 cliff rotations. Wherever any
    # rotation has the bright brown edge, at least one other rotation has the
    # dark interior body at the same pixel, so the min picks dark. Used for
    # interior wall cells with no floor neighbors (e.g., the (8,8) cross
    # intersection in FourRooms) — otherwise we'd see a floating brown lip.
    _DARK_VOID_TILE = ImageChops.darker(
        ImageChops.darker(_CLIFF_ROT["t"], _CLIFF_ROT["b"]),
        ImageChops.darker(_CLIFF_ROT["l"], _CLIFF_ROT["r"]),
    )
    _ENT_TILES = {
        "stool":     _load_entity_fitted("stool.png",     max_dim_scale=1.1),
        "cookie":    _load_entity_fitted("cookie.png",    max_dim_scale=1.25),
        "elf_up":    _load_entity_fitted("elf_up.png",    height_scale=1.6),
        "elf_right": _load_entity_fitted("elf_right.png", height_scale=1.6),
        "elf_down":  _load_entity_fitted("elf_down.png",  height_scale=1.6),
        "elf_left":  _load_entity_fitted("elf_left.png",  height_scale=1.6),
    }


# Fading trail. Magenta-pink pops against green grass (warm/cool contrast)
# and stays distinct from the elf's red shirt and the bump-burst yellow.
TRAIL_COLOR       = (235, 60, 175)
TRAIL_MAX_ALPHA   = 240
TRAIL_MIN_ALPHA   = 110
TRAIL_FADE_FRAMES = 25


# Direction inferred from delta (dy, dx) -> elf sprite key
DELTA_TO_ELF = {
    (-1,  0): "elf_up",
    ( 1,  0): "elf_down",
    ( 0, -1): "elf_left",
    ( 0,  1): "elf_right",
}


# -----------------------------------------------------------------------
# tile / sprite compositing helpers
# -----------------------------------------------------------------------
def _make_wall_tile(walls: np.ndarray, y: int, x: int) -> Image.Image:
    """Build a wall tile by per-pixel-max-merging cliff edges, one rotated
    instance per side that borders a floor cell. Brown rocky edge is
    brighter than the dark void body, so ImageChops.lighter (per-pixel max)
    picks the brown edge wherever any rotation supplies it."""
    _ensure_loaded()
    assert _CLIFF_ROT is not None and _DARK_VOID_TILE is not None
    H, W = walls.shape
    sides: list[str] = []
    if y > 0       and not walls[y - 1, x]: sides.append("t")
    if y < H - 1   and not walls[y + 1, x]: sides.append("b")
    if x > 0       and not walls[y, x - 1]: sides.append("l")
    if x < W - 1   and not walls[y, x + 1]: sides.append("r")
    if not sides:
        return _DARK_VOID_TILE
    result = _CLIFF_ROT[sides[0]]
    for s in sides[1:]:
        result = ImageChops.lighter(result, _CLIFF_ROT[s])
    return result


def _composite_centered(canvas: Image.Image, sprite: Image.Image,
                        cell_y: int, cell_x: int,
                        y_pixel_offset: int = 0) -> None:
    """alpha_composite `sprite` centered on (cell_y, cell_x). Crops the
    source if the sprite would land partially outside `canvas`."""
    sw, sh = sprite.size
    cw, ch = canvas.size
    cx_px = cell_x * CELL + CELL // 2
    cy_px = cell_y * CELL + CELL // 2 + y_pixel_offset
    px = cx_px - sw // 2
    py = cy_px - sh // 2
    src_x0 = max(0, -px)
    src_y0 = max(0, -py)
    src_x1 = min(sw, cw - px)
    src_y1 = min(sh, ch - py)
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return
    cropped = sprite.crop((src_x0, src_y0, src_x1, src_y1))
    canvas.alpha_composite(cropped, (px + src_x0, py + src_y0))


# -----------------------------------------------------------------------
# bg compositing
# -----------------------------------------------------------------------
def build_static_bg(walls: np.ndarray,
                    goal_xy: tuple[int, int],
                    start_xy: tuple[int, int] | None,
                    ) -> Image.Image:
    """Per-episode static background: bg checkerboard + 4-side wall tiles
    + start stool + goal cookie."""
    _ensure_loaded()
    assert _TILES is not None and _ENT_TILES is not None
    H, W = walls.shape
    canvas = Image.new("RGBA", (W * CELL, H * CELL), (0, 0, 0, 255))

    for y in range(H):
        for x in range(W):
            mask = (y % 2) ^ (x % 2)
            tile = _TILES["bg1" if mask else "bg2"]
            canvas.paste(tile, (x * CELL, y * CELL))

    for y in range(H):
        for x in range(W):
            if walls[y, x]:
                canvas.alpha_composite(
                    _make_wall_tile(walls, y, x),
                    (x * CELL, y * CELL),
                )

    if start_xy is not None:
        sy, sx = start_xy
        _composite_centered(canvas, _ENT_TILES["stool"], sy, sx)

    gy, gx = goal_xy
    _composite_centered(canvas, _ENT_TILES["cookie"], gy, gx)

    return canvas


def composite_trail(canvas: Image.Image,
                    states: list[tuple[int, int]],
                    current_i: int) -> Image.Image:
    """Return a copy of canvas with a fading polyline trail drawn through
    cell centers from states[0] to states[current_i]. Newer segments are
    brighter; segments older than TRAIL_FADE_FRAMES dim to TRAIL_MIN_ALPHA
    and stay visible (faint) for the rest of the run."""
    out = canvas.copy()
    if current_i <= 0:
        return out
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = max(2, CELL // 10)
    for j in range(current_i):
        age = current_i - j
        if age >= TRAIL_FADE_FRAMES:
            alpha = TRAIL_MIN_ALPHA
        else:
            t = age / TRAIL_FADE_FRAMES
            alpha = int(TRAIL_MAX_ALPHA * (1.0 - t) + TRAIL_MIN_ALPHA * t)
        if alpha <= 0:
            continue
        y0, x0 = states[j]
        y1, x1 = states[j + 1]
        d.line(
            [(x0 * CELL + CELL // 2, y0 * CELL + CELL // 2),
             (x1 * CELL + CELL // 2, y1 * CELL + CELL // 2)],
            fill=(*TRAIL_COLOR, alpha),
            width=width,
        )
    out.alpha_composite(layer)
    return out


def composite_elf(canvas: Image.Image, state: tuple[int, int],
                  elf_key: str, bump: bool = False) -> Image.Image:
    """Composite elf at the given state on a copy of canvas. Adds a
    yellow/orange burst overlay if ``bump=True`` (i.e. the agent tried to
    move but was blocked by a wall or grid edge on this step)."""
    _ensure_loaded()
    assert _ENT_TILES is not None
    out = canvas.copy()
    y, x = state

    if bump:
        # Yellow burst BEHIND the elf for visibility. Sized larger than CELL
        # so sparkle rays poke out beyond the (1.6x cell) elf silhouette.
        BURST = int(CELL * 1.8)
        burst = Image.new("RGBA", (BURST, BURST), (0, 0, 0, 0))
        d = ImageDraw.Draw(burst)
        c = BURST // 2
        for r, a in [(BURST // 2, 55), (BURST // 3, 100), (BURST // 5, 170)]:
            d.ellipse((c - r, c - r, c + r, c + r),
                      fill=(255, 220, 60, a))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            x1, y1 = c + dx * (BURST // 3), c + dy * (BURST // 3)
            x2, y2 = c + dx * (BURST // 2 - 3), c + dy * (BURST // 2 - 3)
            d.line([(x1, y1), (x2, y2)], fill=(255, 250, 200, 230), width=5)
        _composite_centered(out, burst, y, x)

    # Farama upward offset of -0.1*CELL so elf "stands above" the cell
    _composite_centered(out, _ENT_TILES[elf_key], y, x,
                        y_pixel_offset=-int(0.1 * CELL))
    return out


# -----------------------------------------------------------------------
# direction inference for synthetic trajectories (no actions logged)
# -----------------------------------------------------------------------
def infer_elf_keys(states: list[tuple[int, int]],
                   start_facing: str = "elf_down") -> list[str]:
    """For a list of states, infer which elf sprite to display at each step
    based on the NEXT step's delta. Last step keeps the previous facing.
    Stays (no-move) keep last facing."""
    keys: list[str] = []
    last = start_facing
    for i in range(len(states)):
        if i + 1 < len(states):
            dy = states[i + 1][0] - states[i][0]
            dx = states[i + 1][1] - states[i][1]
            k = DELTA_TO_ELF.get((dy, dx))
            if k is not None:
                last = k
        keys.append(last)
    return keys


# -----------------------------------------------------------------------
# rendering
# -----------------------------------------------------------------------
def render_static(walls: np.ndarray,
                  goal_xy: tuple[int, int],
                  start_xy: tuple[int, int],
                  out_path: str | Path) -> Path:
    """Save a single PNG frame: bg + walls + start + goal + preview elf."""
    out_path = Path(out_path)
    bg = build_static_bg(walls, goal_xy, start_xy)
    preview = composite_elf(bg, start_xy, "elf_down")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(out_path)
    return out_path


def render_traj_gif(walls: np.ndarray,
                    goal_xy: tuple[int, int],
                    start_xy: tuple[int, int],
                    states: list[tuple[int, int]],
                    bump_idx: list[int],
                    out_path: str | Path,
                    title: str = "",
                    fps: int = 6) -> Path:
    """Save a trajectory GIF. ``bump_idx`` lists step indices where the
    agent was blocked by a wall — those frames get the yellow/orange burst
    overlay. Free-movement slips are NOT highlighted."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = walls.shape
    bg = build_static_bg(walls, goal_xy, start_xy)
    elf_keys = infer_elf_keys(states)
    bump_set = set(bump_idx)

    trail0 = composite_trail(bg, states, 0)
    f0 = composite_elf(trail0, states[0], elf_keys[0], bump=(0 in bump_set))
    arr0 = np.array(f0.convert("RGB"))

    fig_w = W * CELL / 110
    fig_h = H * CELL / 110 + 0.5  # tiny margin for title
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=110)
    im = ax.imshow(arr0, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes,
                         ha="center", fontsize=10, fontfamily="monospace",
                         color="#222")
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)

    n = len(states)

    def update(i):
        is_bump = i in bump_set
        trailed = composite_trail(bg, states, i)
        frame = composite_elf(trailed, states[i], elf_keys[i], bump=is_bump)
        im.set_array(np.array(frame.convert("RGB")))
        flag = "    ⚠ BUMP" if is_bump else ""
        title_text.set_text(f"step {i:>3}/{n - 1}{flag}")
        return im, title_text

    anim = animation.FuncAnimation(fig, update, frames=n,
                                   interval=int(1000 / fps),
                                   blit=False, repeat=False)
    anim.save(str(out_path), writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
