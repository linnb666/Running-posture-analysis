from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MARGIN = 72
CARD_RADIUS = 28
BACKGROUND = "#F4EFE7"
SURFACE = "#FFFDF8"
PANEL = "#FFFFFF"
ACCENT = "#B76A24"
TEXT = "#2B221B"
MUTED = "#746557"
BORDER = "#E6D7C7"
SUCCESS = "#6D8E54"
WARNING = "#C0673D"

plt.rcParams["axes.unicode_minus"] = False

MATPLOTLIB_FONT = None
for _candidate in [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]:
    if Path(_candidate).exists():
        MATPLOTLIB_FONT = _candidate
        break

PIL_FONT_REGULAR = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
PIL_FONT_BOLD = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = PIL_FONT_BOLD if bold else PIL_FONT_REGULAR
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _mpl_font(size: int = 12):
    if MATPLOTLIB_FONT:
        return font_manager.FontProperties(fname=MATPLOTLIB_FONT, size=size)
    return None


def _rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill: str, outline: str | None = None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _parse_local_text(value) -> str:
    if not value:
        return "--"
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    try:
        return draw.textlength(text, font=font)
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def _line_height(draw: ImageDraw.ImageDraw, font, line_gap: int = 10) -> int:
    bbox = draw.textbbox((0, 0), "Hg", font=font)
    return (bbox[3] - bbox[1]) + line_gap


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        paragraph = paragraph.rstrip()
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            candidate = current + ch
            if current and _text_width(draw, candidate, font) > max_width:
                lines.append(current)
                current = ch
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines


def _clip_lines(lines: list[str], max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    if len(lines) <= max_lines:
        return lines
    clipped = list(lines[:max_lines])
    if clipped and clipped[-1]:
        clipped[-1] = clipped[-1].rstrip("。；;，, ") + "…"
    return clipped


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    lines: Iterable[str],
    font,
    fill: str,
    line_gap: int = 12,
):
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += _line_height(draw, font, line_gap)
    return y


def _fit_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    height: int,
    sizes: list[int],
    *,
    bold: bool = False,
    line_gap: int = 8,
    max_lines: int | None = None,
):
    best_font = _load_font(sizes[-1], bold=bold)
    best_lines = _clip_lines(_wrap_text(draw, text, best_font, width), max_lines or 999)
    for size in sizes:
        font = _load_font(size, bold=bold)
        lines = _wrap_text(draw, text, font, width)
        if max_lines is not None:
            lines = _clip_lines(lines, max_lines)
        required = max(1, len(lines)) * _line_height(draw, font, line_gap)
        if required <= height:
            return font, lines
        best_font, best_lines = font, lines
    return best_font, best_lines


def _draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill: str,
    line_gap: int = 10,
    max_lines: int | None = None,
):
    x0, y0, x1, y1 = box
    lines = _wrap_text(draw, text, font, x1 - x0)
    if max_lines is None:
        max_lines = max(1, (y1 - y0) // max(1, _line_height(draw, font, line_gap)))
    lines = _clip_lines(lines, max_lines)
    return _draw_lines(draw, (x0, y0), lines, font, fill, line_gap=line_gap)


def _fit_image(path: str | Path, width: int, height: int) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return ImageOps.contain(img, (width, height))
    except Exception:
        return None


def _fit_image_cover(path: str | Path, width: int, height: int, focus_y: float = 0.42) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            src_ratio = img.width / max(1, img.height)
            dst_ratio = width / max(1, height)
            if src_ratio > dst_ratio:
                scaled_h = height
                scaled_w = int(img.width * (height / max(1, img.height)))
                img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                left = max(0, int((scaled_w - width) / 2))
                return img.crop((left, 0, left + width, height))
            scaled_w = width
            scaled_h = int(img.height * (width / max(1, img.width)))
            img = img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            top = int((scaled_h - height) * max(0.0, min(1.0, focus_y)))
            top = max(0, min(top, max(0, scaled_h - height)))
            return img.crop((0, top, width, top + height))
    except Exception:
        return None


def _new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(page)
    return page, draw


def _to_image(buffer: BytesIO) -> Image.Image:
    buffer.seek(0)
    with Image.open(buffer) as img:
        return img.convert("RGBA")


def _chart_canvas(fig) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="#fffdf9")
    plt.close(fig)
    return _to_image(buffer)


def _create_radar_chart(dimension_scores: dict[str, float]) -> Image.Image:
    items = list((dimension_scores or {}).items())[:6]
    if not items:
        fig = plt.figure(figsize=(5.0, 4.1), dpi=180)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "暂无维度评分数据", ha="center", va="center", fontsize=18, color="#746557", fontproperties=_mpl_font(14))
        return _chart_canvas(fig)

    labels = [str(item[0]) for item in items]
    values = [float(item[1] or 0) for item in items]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_cycle = values + values[:1]
    angles_cycle = angles + angles[:1]

    fig = plt.figure(figsize=(5.2, 4.3), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#fffdf9")
    ax.plot(angles_cycle, values_cycle, color=ACCENT, linewidth=2.6)
    ax.fill(angles_cycle, values_cycle, color="#EFDCC8", alpha=0.72)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#8A7B6B", fontsize=9)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    font_prop = _mpl_font(11)
    for label in ax.get_xticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)
        label.set_fontsize(11)
        label.set_color(TEXT)
    ax.grid(color="#E6D7C7", linewidth=1)
    ax.spines["polar"].set_color("#D6C0A5")
    return _chart_canvas(fig)


def _create_dimension_bar_chart(dimension_scores: dict[str, float]) -> Image.Image:
    items = list((dimension_scores or {}).items())[:6]
    if not items:
        fig = plt.figure(figsize=(5.0, 4.1), dpi=180)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "暂无维度评分数据", ha="center", va="center", fontsize=18, color="#746557", fontproperties=_mpl_font(14))
        return _chart_canvas(fig)

    labels = [str(item[0]) for item in items]
    values = [float(item[1] or 0) for item in items]
    palette = [ACCENT, "#C9884A", "#D9A86C", "#8FA96C", "#B0BCA0", "#D2C2AF"]

    fig = plt.figure(figsize=(5.2, 4.3), dpi=180)
    ax = fig.add_subplot(111)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=palette[: len(values)], edgecolor="#E1CFB9", height=0.58)
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    font_prop = _mpl_font(11)
    for label in ax.get_yticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)
        label.set_fontsize(11)
        label.set_color(TEXT)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, colors="#8A7B6B")
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#EFE3D4", linewidth=1)
    ax.set_axisbelow(True)
    for idx, value in enumerate(values):
        ax.text(min(value + 2.0, 97.0), idx, f"{value:.1f}", va="center", ha="left", fontsize=10, color=TEXT)
    return _chart_canvas(fig)


def _paste_chart(page: Image.Image, chart: Image.Image, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    image = ImageOps.contain(chart, (x1 - x0, y1 - y0))
    px = x0 + int((x1 - x0 - image.width) / 2)
    py = y0 + int((y1 - y0 - image.height) / 2)
    page.paste(image, (px, py), image)


def _metric_rows(items: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    count = len(items)
    if count <= 2:
        return [items]
    if count == 3:
        return [items]
    if count == 4:
        return [items[:2], items[2:]]
    if count == 5:
        return [items[:3], items[3:]]
    return [items[:3], items[3:6]]


def _metric_box_height(metrics: list[tuple[str, str]]) -> int:
    rows = _metric_rows((metrics or [])[:6])
    if not rows:
        return 132
    return 68 + len(rows) * 102 + max(0, len(rows) - 1) * 16 + 28


def _draw_metric_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], metrics: list[tuple[str, str]]):
    title_font = _load_font(22, bold=True)
    label_font = _load_font(18)
    x0, y0, x1, y1 = box
    _rounded(draw, box, 24, fill=PANEL, outline=BORDER, width=2)
    draw.text((x0 + 24, y0 + 20), "核心指标概览", font=title_font, fill=TEXT)

    items = metrics[:6]
    if not items:
        draw.text((x0 + 24, y0 + 74), "暂无核心指标", font=label_font, fill=MUTED)
        return

    rows = _metric_rows(items)
    inner_left = x0 + 24
    inner_top = y0 + 68
    inner_width = x1 - x0 - 48
    row_gap = 16
    card_h = 102
    y = inner_top
    for row_items in rows:
        cols = len(row_items)
        col_gap = 18
        row_width = inner_width if cols == 3 else int(inner_width * 0.78)
        card_w = int((row_width - col_gap * (cols - 1)) / cols)
        start_x = inner_left + int((inner_width - row_width) / 2)
        for idx, (label, value) in enumerate(row_items):
            card_x = start_x + idx * (card_w + col_gap)
            card_box = (card_x, y, card_x + card_w, y + card_h)
            _rounded(draw, card_box, 18, fill="#FCF9F4", outline="#EFE2D3", width=1)
            draw.text((card_x + 14, y + 14), str(label), font=label_font, fill=MUTED)
            value_font, value_lines = _fit_wrapped_lines(draw, str(value), card_w - 28, 48, [28, 26, 24, 22, 20], bold=True, line_gap=4, max_lines=2)
            _draw_lines(draw, (card_x + 14, y + 42), value_lines, value_font, TEXT, line_gap=4)
        y += card_h + row_gap

def _draw_summary_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, title_color: str):
    title_font = _load_font(22, bold=True)
    x0, y0, x1, y1 = box
    _rounded(draw, box, 22, fill=PANEL, outline=BORDER, width=2)
    draw.text((x0 + 20, y0 + 18), title, font=title_font, fill=title_color)
    text_height = y1 - y0 - 86
    body_font, lines = _fit_wrapped_lines(draw, body or "暂无", x1 - x0 - 40, text_height, [18, 17, 16, 15], line_gap=7)
    max_lines = max(1, text_height // max(1, _line_height(draw, body_font, 7)))
    lines = _clip_lines(lines, max_lines)
    _draw_lines(draw, (x0 + 20, y0 + 56), lines, body_font, MUTED, line_gap=7)


def _draw_footer(draw: ImageDraw.ImageDraw, page_number: int, total_pages: int):
    font = _load_font(18)
    left = "跑步动作分析系统 / PDF 报告"
    right = f"第 {page_number} / {total_pages} 页"
    draw.text((MARGIN, PAGE_HEIGHT - 44), left, font=font, fill="#8A7B6B")
    width = _text_width(draw, right, font)
    draw.text((PAGE_WIDTH - MARGIN - int(width), PAGE_HEIGHT - 44), right, font=font, fill="#8A7B6B")


def _render_cover_page(data: dict) -> Image.Image:
    page, draw = _new_page()
    small_font = _load_font(20)
    body_font = _load_font(24)
    hero_value_font = _load_font(56, bold=True)
    section_font = _load_font(24, bold=True)

    _rounded(draw, (MARGIN, 52, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 70), 36, fill=SURFACE, outline=BORDER, width=2)
    hero_bottom = 362
    _rounded(draw, (MARGIN + 22, 80, PAGE_WIDTH - MARGIN - 22, hero_bottom), 32, fill="#F8E8D5", outline="#E8D1B7", width=2)

    draw.text((MARGIN + 54, 114), "Running Analysis Report", font=small_font, fill=ACCENT)
    title_font, title_lines = _fit_wrapped_lines(draw, str(data.get("video_filename") or "--"), 560, 90, [44, 40, 36, 32, 30], bold=True, line_gap=6, max_lines=2)
    title_y = _draw_lines(draw, (MARGIN + 54, 146), title_lines, title_font, TEXT, line_gap=6)
    draw.text((MARGIN + 54, title_y + 6), "跑步动作分析系统导出报告", font=body_font, fill=MUTED)

    badge_box = (PAGE_WIDTH - MARGIN - 266, 116, PAGE_WIDTH - MARGIN - 40, 306)
    _rounded(draw, badge_box, 26, fill="#FFFDF8", outline="#E4CBAF", width=2)
    draw.text((badge_box[0] + 28, badge_box[1] + 22), "综合得分", font=small_font, fill=MUTED)
    draw.text((badge_box[0] + 28, badge_box[1] + 54), f"{float(data.get('total_score') or 0):.1f}", font=hero_value_font, fill=ACCENT)
    draw.text((badge_box[0] + 28, badge_box[1] + 126), str(data.get("rating") or "--"), font=body_font, fill=TEXT)

    meta_x = MARGIN + 54
    meta_y = title_y + 44
    meta_width = badge_box[0] - meta_x - 24
    for label, value in [
        ("任务 ID", data.get("task_id") or "--"),
        ("分析视角", data.get("view_label") or "--"),
        ("分析时间", _parse_local_text(data.get("created_at"))),
        ("模型版本", str(data.get("model_version") or "N/A")),
    ]:
        meta_font, meta_lines = _fit_wrapped_lines(draw, f"{label}: {value}", meta_width, 42, [20, 19, 18, 17], line_gap=4, max_lines=2)
        meta_y = _draw_lines(draw, (meta_x, meta_y), meta_lines, meta_font, MUTED, line_gap=4)
        meta_y += 8

    chart_top = 394
    left_card = (MARGIN + 24, chart_top, PAGE_WIDTH // 2 - 16, 884)
    right_card = (PAGE_WIDTH // 2 + 16, chart_top, PAGE_WIDTH - MARGIN - 24, 884)
    for box, title in [(left_card, "维度雷达图"), (right_card, "维度条形图")]:
        _rounded(draw, box, 26, fill=PANEL, outline=BORDER, width=2)
        draw.text((box[0] + 22, box[1] + 18), title, font=section_font, fill=TEXT)

    radar = _create_radar_chart(data.get("dimension_scores") or {})
    bars = _create_dimension_bar_chart(data.get("dimension_scores") or {})
    _paste_chart(page, radar, (left_card[0] + 10, left_card[1] + 56, left_card[2] - 10, left_card[3] - 16))
    _paste_chart(page, bars, (right_card[0] + 10, right_card[1] + 56, right_card[2] - 10, right_card[3] - 16))

    metric_top = 914
    metric_height = _metric_box_height(data.get("metrics") or [])
    metric_box = (MARGIN + 24, metric_top, PAGE_WIDTH - MARGIN - 24, metric_top + metric_height)
    _draw_metric_grid(draw, metric_box, data.get("metrics") or [])

    summary_top = metric_box[3] + 28
    gap = 18
    card_width = int((PAGE_WIDTH - MARGIN * 2 - 48 - gap * 2) / 3)
    strengths = "；".join(data.get("strengths") or []) or "暂无明显优势描述"
    weaknesses = "；".join(data.get("weaknesses") or []) or "暂无明显薄弱项"
    suggestions = "；".join(data.get("suggestions") or []) or "暂无训练建议"
    cards = [
        ("已识别优势", strengths, SUCCESS),
        ("待改进项", weaknesses, WARNING),
        ("训练建议", suggestions, ACCENT),
    ]
    for idx, (title, content, color) in enumerate(cards):
        x0 = MARGIN + 24 + idx * (card_width + gap)
        box = (x0, summary_top, x0 + card_width, 1602)
        _draw_summary_card(draw, box, title, content, color)

    return page

def _render_keyframes_page(data: dict) -> Image.Image:
    page, draw = _new_page()
    title_font = _load_font(36, bold=True)
    body_font = _load_font(22)
    small_font = _load_font(18)

    _rounded(draw, (MARGIN, 52, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 70), 36, fill=SURFACE, outline=BORDER, width=2)
    draw.text((MARGIN + 28, 86), "关键帧与人工备注", font=title_font, fill=TEXT)
    draw.text((MARGIN + 28, 132), "用于结果复核、论文插图和人工补充说明。", font=body_font, fill=MUTED)

    grid_top = 188
    card_w = int((PAGE_WIDTH - MARGIN * 2 - 84) / 2)
    card_h = 352
    image_h = 238
    is_front_view = "正面" in str(data.get("view_label") or "")
    keyframes = data.get("keyframes") or []
    for idx in range(4):
        row = idx // 2
        col = idx % 2
        x0 = MARGIN + 28 + col * (card_w + 28)
        y0 = grid_top + row * (card_h + 24)
        x1 = x0 + card_w
        y1 = y0 + card_h
        _rounded(draw, (x0, y0, x1, y1), 24, fill=PANEL, outline=BORDER, width=2)
        image_box = (x0 + 14, y0 + 14, x1 - 14, y0 + 14 + image_h)
        if idx < len(keyframes):
            item = keyframes[idx]
            if is_front_view:
                image = _fit_image(item.get("path", ""), image_box[2] - image_box[0], image_box[3] - image_box[1])
            else:
                image = _fit_image_cover(item.get("path", ""), image_box[2] - image_box[0], image_box[3] - image_box[1], focus_y=0.35)
                if image is None:
                    image = _fit_image(item.get("path", ""), image_box[2] - image_box[0], image_box[3] - image_box[1])
            if image is not None:
                px = image_box[0] + int((image_box[2] - image_box[0] - image.width) / 2)
                py = image_box[1] + int((image_box[3] - image_box[1] - image.height) / 2)
                page.paste(image, (px, py))
            label = str(item.get("label") or "关键帧")
            draw.text((x0 + 18, y0 + image_h + 34), label, font=small_font, fill=MUTED)
        else:
            draw.text((x0 + 18, y0 + 22), "暂无关键帧", font=small_font, fill=MUTED)

    notes_box = (MARGIN + 28, 976, PAGE_WIDTH - MARGIN - 28, 1588)
    _rounded(draw, notes_box, 26, fill=PANEL, outline=BORDER, width=2)
    draw.text((notes_box[0] + 22, notes_box[1] + 20), "人工备注", font=_load_font(24, bold=True), fill=TEXT)
    meta_text = _parse_local_text(data.get("manual_notes_updated_at")) if data.get("manual_notes_updated_at") else "未单独保存"
    draw.text((notes_box[0] + 22, notes_box[1] + 58), f"最近保存：{meta_text}", font=small_font, fill=MUTED)
    notes = (data.get("manual_notes") or "").strip() or "暂无人工备注"
    left_col = (notes_box[0] + 22, notes_box[1] + 104, notes_box[0] + 560, notes_box[3] - 22)
    _draw_wrapped_block(draw, left_col, notes, body_font, MUTED, line_gap=8)

    right_panel = (notes_box[0] + 596, notes_box[1] + 92, notes_box[2] - 22, notes_box[3] - 22)
    _rounded(draw, right_panel, 18, fill="#FCF9F4", outline="#EFE2D3", width=1)
    draw.text((right_panel[0] + 16, right_panel[1] + 16), "报告摘要", font=_load_font(22, bold=True), fill=TEXT)
    summary_items = [
        ("综合得分", f"{float(data.get('total_score') or 0):.1f}"),
        ("评级", str(data.get("rating") or "--")),
        ("视角", str(data.get("view_label") or "--")),
        ("模型版本", str(data.get("model_version") or "N/A")),
    ]
    sy = right_panel[1] + 58
    for label, value in summary_items:
        draw.text((right_panel[0] + 16, sy), label, font=small_font, fill=MUTED)
        value_font, value_lines = _fit_wrapped_lines(draw, value, right_panel[2] - right_panel[0] - 32, 42, [24, 22, 20, 18], bold=True, line_gap=4, max_lines=2)
        _draw_lines(draw, (right_panel[0] + 16, sy + 22), value_lines, value_font, TEXT, line_gap=4)
        sy += 98

    return page

def _clean_markdown_inline(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    return cleaned.strip()


def _prepare_report_blocks(report_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw in str(report_text or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line:
            blocks.append(("blank", ""))
            continue
        stripped = line.lstrip()
        if stripped.startswith("###"):
            blocks.append(("heading3", _clean_markdown_inline(stripped.lstrip("# "))))
        elif stripped.startswith("##"):
            blocks.append(("heading2", _clean_markdown_inline(stripped.lstrip("# "))))
        elif stripped.startswith("#"):
            blocks.append(("heading1", _clean_markdown_inline(stripped.lstrip("# "))))
        elif re.match(r"^\d+\.\s+", stripped):
            blocks.append(("ordered", _clean_markdown_inline(stripped)))
        elif stripped.startswith(("- ", "* ")):
            blocks.append(("bullet", _clean_markdown_inline(stripped[2:].strip())))
        else:
            blocks.append(("paragraph", _clean_markdown_inline(stripped)))
    compact: list[tuple[str, str]] = []
    previous_blank = False
    for kind, inner_text in blocks:
        if kind == "blank":
            if previous_blank:
                continue
            previous_blank = True
        else:
            previous_blank = False
        compact.append((kind, inner_text))
    return compact or [("paragraph", "暂无报告内容")]

def _paginate_report_blocks(draw: ImageDraw.ImageDraw, blocks: list[tuple[str, str]], available_height: int) -> list[list[tuple[str, str]]]:
    title_font = _load_font(32, bold=True)
    heading_font = _load_font(23, bold=True)
    body_font = _load_font(20)

    def block_height(kind: str, text: str) -> int:
        if kind == "blank":
            return 16
        if kind == "heading1":
            font, gap, indent = title_font, 12, 0
        elif kind in {"heading2", "heading3"}:
            font, gap, indent = heading_font, 10, 0
        elif kind == "bullet":
            font, gap, indent = body_font, 6, 24
            text = f"• {text}"
        elif kind == "ordered":
            font, gap, indent = body_font, 6, 20
        else:
            font, gap, indent = body_font, 6, 0
        lines = _wrap_text(draw, text, font, PAGE_WIDTH - MARGIN * 2 - 84 - indent)
        return len(lines) * _line_height(draw, font, 8) + gap

    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_height = 0
    heights: list[int] = []
    for block in blocks:
        h = block_height(*block)
        if current and current_height + h > available_height:
            pages.append(current)
            heights.append(current_height)
            current = [block]
            current_height = h
        else:
            current.append(block)
            current_height += h
    if current:
        pages.append(current)
        heights.append(current_height)

    if len(pages) >= 2 and heights[-1] < int(available_height * 0.22):
        previous = pages[-2]
        last = pages[-1]
        prev_height = heights[-2]
        last_height = heights[-1]
        while len(previous) > 1:
            candidate = previous[-1]
            h = block_height(*candidate)
            if last_height + h > int(available_height * 0.48):
                break
            previous.pop()
            last.insert(0, candidate)
            prev_height -= h
            last_height += h
            if prev_height < int(available_height * 0.70):
                break
        if previous:
            pages[-2] = previous
            pages[-1] = last
    return pages

def _render_report_pages(data: dict) -> list[Image.Image]:
    pages: list[Image.Image] = []
    title_font = _load_font(32, bold=True)
    heading_font = _load_font(23, bold=True)
    body_font = _load_font(20)
    small_font = _load_font(18)

    probe_page, probe_draw = _new_page()
    del probe_page
    content_top = 182
    content_bottom = PAGE_HEIGHT - 118
    paged_blocks = _paginate_report_blocks(probe_draw, _prepare_report_blocks(data.get("analysis_text") or "暂无报告内容"), content_bottom - content_top)

    for page_blocks in paged_blocks:
        page, draw = _new_page()
        _rounded(draw, (MARGIN, 52, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 70), 36, fill=SURFACE, outline=BORDER, width=2)
        draw.text((MARGIN + 30, 86), str(data.get("analysis_source") or "分析报告"), font=title_font, fill=TEXT)
        draw.text((MARGIN + 30, 128), "以下内容用于结果归档与后续复查。", font=small_font, fill=MUTED)
        y = content_top
        for kind, inner_text in page_blocks:
            if kind == "blank":
                y += 16
                continue
            if kind == "heading1":
                font = title_font
                fill = TEXT
                indent = 0
                gap = 12
            elif kind in {"heading2", "heading3"}:
                font = heading_font
                fill = TEXT
                indent = 0
                gap = 10
            elif kind == "bullet":
                font = body_font
                fill = MUTED
                indent = 24
                inner_text = f"• {inner_text}"
                gap = 6
            elif kind == "ordered":
                font = body_font
                fill = MUTED
                indent = 20
                gap = 6
            else:
                font = body_font
                fill = MUTED
                indent = 0
                gap = 6
            lines = _wrap_text(draw, inner_text, font, PAGE_WIDTH - MARGIN * 2 - 84 - indent)
            y = _draw_lines(draw, (MARGIN + 30 + indent, y), lines, font, fill, line_gap=8)
            y += gap
        pages.append(page)
    return pages

def build_report_pdf(report_data: dict) -> bytes:
    pages = [
        _render_cover_page(report_data),
        _render_keyframes_page(report_data),
        *_render_report_pages(report_data),
    ]
    total_pages = len(pages)
    for page_number, page in enumerate(pages, start=1):
        draw = ImageDraw.Draw(page)
        _draw_footer(draw, page_number, total_pages)
    rgb_pages = [page.convert("RGB") for page in pages]
    buffer = BytesIO()
    rgb_pages[0].save(buffer, format="PDF", resolution=150.0, save_all=True, append_images=rgb_pages[1:])
    return buffer.getvalue()


