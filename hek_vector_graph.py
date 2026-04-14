#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hek_vector_graph.py  –  SVG vector / graph drawing library
===========================================================
Depends on:
    py2html  – Builder / SELFCLOSING helpers
    hek_map_utils – Vector, Position namedtuples

Improvements over the original
-------------------------------
* Fixed: xml scoping in draw_line (was relying on global `xml`, now stored as self.xml)
* Fixed: Circle referenced bare `colors` instead of `G.colors`
* Fixed: graph() "cm" check now guards against non-string width/height
* Fixed: G.colors['dark'] was the literal string 'dark' in dark-mode; now a real colour
* Fixed: Label / DrawText passed non-string text to len() – now uses str()
* Fixed: Label.__init__ used `return ""` (no-op in __init__) – changed to plain return
* Fixed: ForeignObject signature mismatch – accepts Position OR (px, py) pair
* Fixed: Polygon.sides – guard against already-split list and empty list
* Fixed: dashed=False in Angle passed False to Line which left it as a literal SVG value
* Implemented: Plot class – rewrote to use np.linspace and accept callables as well as strings
* Implemented: LeaderLine JavaScript connector stub (draws visible lines + JS positioning)
* Improved: Arrow draws correctly toward the exact `to` point by computing px2/py2 via y_to_py
* Improved: dark-mode colour palette (was using 'dark' as a colour name)
* Improved: setup_the_axes now conditionally calls draw_grid only when grid_step_size > 0
"""

import numpy as np
import math
from py2html import Builder, SELFCLOSING
from hek_map_utils import Vector, Position

# ---------------------------------------------------------------------------
# Coordinate helpers  (module-level lambdas kept for backward-compat)
# ---------------------------------------------------------------------------
x_to_px = lambda x: (x - G.x_min) * G.xScale
y_to_py = lambda y: G.height - (y - G.y_min) * G.yScale

CM = 37.79527559  # pixels per centimetre

# ---------------------------------------------------------------------------
# Global configuration class
# ---------------------------------------------------------------------------
class G:
    sizes       = {'tiny': 8,  'small': 8,  'normal': 10, 'large': 12}
    fontsizes   = {'tiny': 12, 'small': 12, 'normal': 16, 'large': 20}
    strokesizes = {'tiny': 0,  'small': 1,  'normal': 1.75, 'large': 2}
    global_id   = 0

# ---------------------------------------------------------------------------
# Graph setup
# ---------------------------------------------------------------------------
def graph(xml, width=200, height=200, pad=30,
          x_min=0, x_max=10, y_min=0, y_max=10,
          grid_step_size=1, dark=False,
          axes=("x", "y"), units=True, id="PLACEHOLDER"):
    """Create an SVG viewport and configure G.*  coordinate transform globals."""
    # FIX: guard "cm" check against non-string arguments
    if isinstance(width, str) and "cm" in width:
        width = int(width[:-2]) * CM
    G.width = width

    if isinstance(height, str) and "cm" in height:
        height = int(height[:-2]) * CM
    G.height = height

    G.x_min        = x_min
    G.x_max        = x_max
    G.y_min        = y_min
    G.y_max        = y_max
    G.grid_step_size = grid_step_size
    G.axes         = axes
    G.units        = units
    G.xScale       = width  / (x_max - x_min)
    G.yScale       = height / (y_max - y_min)
    G.pad          = pad

    # FIX: 'dark' colour value was the literal string 'dark' in the original
    G.colors = {
        'light': '#1a1a1a'  if dark else 'white',
        'gray':  '#555'     if dark else '#ccc',
        'dark':  '#bbb'     if dark else '#555',   # was 'dark' (invalid CSS)
        'black': 'white'    if dark else '#000',
    }

    return xml.svg(
        id       = id,
        width    = f"{width}",
        height   = f"{height}",
        viewbox  = f"{-pad} {-pad * 0.6} {width + 1.6 * pad} {height + 1.6 * pad}",
        xmlns    = "http://www.w3.org/2000/svg",
        style    = f"background: {G.colors['light']}; border-radius: 8px;",
    )

# ---------------------------------------------------------------------------
# Foreign-object helper
# ---------------------------------------------------------------------------
def ForeignObject(xml, position_or_px, py_or_width=None, width_or_height=None, height=None):
    """
    Accepts two calling conventions:
        ForeignObject(xml, Position(px, py), width, height)   ← position object
        ForeignObject(xml, px, py, width, height)              ← explicit px / py

    FIX: original definition required 5 args but was called with 4 (Position + w + h).
    """
    if height is None:
        # Called as (xml, Position, width, height)
        px, py   = position_or_px
        width    = py_or_width
        height   = width_or_height
    else:
        # Called as (xml, px, py, width, height)
        px       = position_or_px
        py       = py_or_width
        width    = width_or_height

    return xml.foreignObject(x=px, y=py, width=width, height=height)

# ---------------------------------------------------------------------------
# Leader / connection line  (drawn visibly; JS positions endpoints at runtime)
# ---------------------------------------------------------------------------
LEADER_LINE_JS = """
<script>
(function () {
    'use strict';
    function getSlotPoint(nodeEl, slot) {
        var r  = nodeEl.getBoundingClientRect();
        var svgR = nodeEl.closest('svg').getBoundingClientRect();
        var cx = r.left + r.width  / 2 - svgR.left;
        var cy = r.top  + r.height / 2 - svgR.top;
        var hw = r.width  / 2, hh = r.height / 2;
        var pts = {
            top:    [cx,      cy - hh],
            bottom: [cx,      cy + hh],
            left:   [cx - hw, cy     ],
            right:  [cx + hw, cy     ],
        };
        return pts[slot] || [cx, cy];
    }

    function updateLeaderLines() {
        document.querySelectorAll('line.connection').forEach(function (line) {
            var startId   = line.getAttribute('start_node');
            var endId     = line.getAttribute('end_node');
            var startSlot = line.getAttribute('start_slot') || 'right';
            var endSlot   = line.getAttribute('end_slot')   || 'left';

            var startEl = document.getElementById(startId);
            var endEl   = document.getElementById(endId);
            if (!startEl || !endEl) return;

            var sp = getSlotPoint(startEl, startSlot);
            var ep = getSlotPoint(endEl,   endSlot);

            line.setAttribute('x1', sp[0]);
            line.setAttribute('y1', sp[1]);
            line.setAttribute('x2', ep[0]);
            line.setAttribute('y2', ep[1]);
            line.setAttribute('visibility', 'visible');
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateLeaderLines);
    } else {
        updateLeaderLines();
    }
    window.addEventListener('resize', updateLeaderLines);
})();
</script>
"""

class LeaderLine:
    """
    Draws an SVG <line> tagged with connection metadata.
    JavaScript (LEADER_LINE_JS) repositions the endpoints at load time
    by looking up start_node / end_node element IDs and the named slot
    (top | bottom | left | right).

    Usage:
        LeaderLine(xml, start_node="box-a", end_node="box-b",
                   start_slot="right", end_slot="left", stroke="black")
    """
    def __init__(self, xml, start_node, end_node,
                 start_slot="right", end_slot="left", **options):
        options.setdefault("stroke",       G.colors['black'])
        options.setdefault("stroke-width", "1.5")
        xml.line(
            visibility  = "hidden",
            class_      = "connection",
            start_node  = start_node,
            end_node    = end_node,
            start_slot  = start_slot,
            end_slot    = end_slot,
            x1=0, y1=0, x2=0, y2=0,   # JS will set real coordinates
            **options,
        )

# ---------------------------------------------------------------------------
# Angle arc
# ---------------------------------------------------------------------------
class Angle:
    def __init__(self, xml, position, label,
                 from_=0, to=0, radius=None,
                 color=None, fontsize=None, dashed=None):
        x, y = self.position = position

        def draw_arc(x, y, r, from_, to):
            def to_euclidean(x, y, radius, rads):
                return [int(x + radius * math.cos(rads)),
                        int(y + radius * math.sin(rads))]
            from_rads = (-from_ * math.pi) / 180
            to_rads   = (-to   * math.pi) / 180
            large     = "0" if to - from_ <= 180 else "1"
            x_start, y_start = to_euclidean(x, y, r, from_rads)
            x_end,   y_end   = to_euclidean(x, y, r, to_rads)
            return f"M {x_start} {y_start} A {int(r)} {int(r)} 0 {large} 0 {x_end} {y_end}"

        if radius is None:
            radius = (G.x_max - G.x_min) / 3
        if color is None:
            color = G.colors['black']
        if fontsize is None:
            fontsize = "small"
        # FIX: dashed=False should produce no dash-array; original left False in SVG attr
        if dashed is None:
            dash_attr = "5,3"
        elif dashed is False:
            dash_attr = "none"
        else:
            dash_attr = str(dashed)

        from_angle = np.deg2rad(from_)
        to_angle   = np.deg2rad(to)

        # Label bisector angle
        if abs(to_angle - from_angle) > np.pi:
            label_angle = (from_angle + to_angle) / 2 + np.pi  # FIX: was missing /2
        else:
            label_angle = (from_angle + to_angle) / 2

        to_angle_deg   = np.rad2deg(to_angle)
        from_angle_deg = np.rad2deg(from_angle)

        if from_angle_deg >= 180: from_angle_deg -= 360
        if to_angle_deg   >= 180: to_angle_deg   -= 360

        from_angle_deg = int(from_angle_deg) % 360
        to_angle_deg   = int(to_angle_deg)   % 360

        px, py = x_to_px(x), y_to_py(y)
        xml.path(
            fill   = "none",
            stroke = color,
            d      = draw_arc(px, py, radius * G.xScale, from_angle_deg, to_angle_deg),
            **{"stroke-dasharray": dash_attr, "stroke-width": "1"},
        )

        # Label position – account for non-square aspect ratio
        if G.xScale == G.yScale:
            x_offset = radius * np.cos(label_angle)
            y_offset = radius * np.sin(label_angle)
        else:
            x_offset = radius * np.cos(label_angle) * G.xScale / G.yScale
            y_offset = radius * np.sin(label_angle)

        Label(xml, text=label,
              position=Position(x + x_offset, y + y_offset),
              fontsize=fontsize, color="blue")

# ---------------------------------------------------------------------------
# Polygon
# ---------------------------------------------------------------------------
class Polygon:
    def __init__(self, xml, points, sides=[], angles=[], color=None):
        self.points = points
        if color is None:
            color = G.colors["black"]

        # FIX: guard against already-split list and empty value
        if isinstance(sides, str):
            sides = sides.split(",") if sides else []

        for i in range(len(points)):
            from_point = points[i]
            to_point   = points[(i + 1) % len(points)]
            label      = sides[i] if sides and i < len(sides) else None
            # FIX: polygon sides are solid, not dashed
            Line(xml, from_=from_point, to=to_point, label=label, color=color,
                 dashed=False)

        if angles:
            if isinstance(angles, str):
                angles = angles.split(",") if angles else []
            for i in range(len(points)):
                from_point = points[i]
                next_point = points[(i + 1) % len(points)]
                prev_point = points[(len(points) + i - 1) % len(points)]
                size       = math.sqrt(
                    (from_point[0] - next_point[0]) ** 2 +
                    (from_point[1] - next_point[1]) ** 2
                )
                next_vec = [next_point[0] - from_point[0],
                            next_point[1] - from_point[1]]
                prev_vec = [from_point[0] - prev_point[0],
                            from_point[1] - prev_point[1]]

                prev_angle = round(
                    180 + (math.atan2(prev_vec[1], prev_vec[0]) * 180) / math.pi
                )
                next_angle = round(
                    (360 + (math.atan2(next_vec[1], next_vec[0]) * 180) / math.pi) % 360
                )

                # FIX: always trace the interior arc.
                # When the naïve span exceeds 180 deg the arc would sweep the
                # exterior instead; swapping from_/to flips it to the interior.
                span = (next_angle - prev_angle) % 360
                if span > 180:
                    prev_angle, next_angle = next_angle, prev_angle

                Angle(xml,
                      position = from_point,
                      label    = angles[i],
                      from_    = prev_angle,
                      to       = next_angle,
                      radius   = size / 3,
                      color    = color)

# ---------------------------------------------------------------------------
# Line
# ---------------------------------------------------------------------------
class Line:
    def __init__(self, xml, to, from_=Position(0, 0),
                 label=None, width=1.75, dashed=True, color=None, axis=False):
        self.xml  = xml   # FIX: store xml so draw_line doesn't rely on global
        self.to   = to
        self.fr   = from_
        if color is None:
            color = G.colors['black']
        # FIX: dashed=False → no dasharray; dashed=True → default pattern
        if dashed is True:
            dashed = "5,3"
        elif dashed is False:
            dashed = None
        self._draw_line(width, color, dashed)
        if label:
            label_x = (to[0] + from_[0]) / 2
            label_y = (to[1] + from_[1]) / 2
            Label(xml, text=label, color=color,
                  position=Position(label_x, label_y))
        if axis:
            Coordinates(xml, position=to, color=color)

    def _draw_line(self, width, color, dashed):
        px1, py1 = x_to_px(self.fr[0]), y_to_py(self.fr[1])
        px2, py2 = x_to_px(self.to[0]), y_to_py(self.to[1])
        attrs = {"x1": px1, "y1": py1, "x2": px2, "y2": py2,
                 "stroke": color, "stroke-width": width}
        if dashed:
            attrs["stroke-dasharray"] = dashed
        self.xml.line(**attrs)

    # Keep old name as an alias for subclass compatibility
    def draw_line(self, width, color, dashed):
        self._draw_line(width, color, dashed)

# ---------------------------------------------------------------------------
# Arrow  (subclasses Line, overrides _draw_line)
# ---------------------------------------------------------------------------
class Arrow(Line):
    def _draw_line(self, width, color, dashed):
        """
        FIX: now uses y_to_py for the final endpoint so the arrowhead lands
        exactly on `to` regardless of scale.  Original computed py2 manually
        and could drift when xScale ≠ yScale.
        """
        px1, py1 = x_to_px(self.fr[0]), y_to_py(self.fr[1])
        px2_full, py2_full = x_to_px(self.to[0]), y_to_py(self.to[1])

        dx = px2_full - px1
        dy = py2_full - py1
        mag   = math.hypot(dx, dy)
        if mag == 0:
            return
        arr_size = 10
        new_mag  = max(0, mag - 2 * arr_size)
        ux, uy   = dx / mag, dy / mag   # unit vector in pixel space

        px2 = px1 + new_mag * ux
        py2 = py1 + new_mag * uy

        global_id    = G.global_id
        G.global_id += 1
        with self.xml.defs:
            with self.xml.marker(id=f"h-{global_id}",
                                 markerWidth="10", markerHeight="5",
                                 refY="2.5", orient="auto"):
                self.xml.polygon(points="0 0, 10 2.5, 0 5", fill=color)

        attrs = {"x1": px1, "y1": py1, "x2": px2, "y2": py2,
                 "stroke": color, "stroke-width": width,
                 "marker-end": f"url(#h-{global_id})"}
        if dashed:
            attrs["stroke-dasharray"] = dashed
        self.xml.line(**attrs)

    # Legacy alias
    def draw_line(self, width, color, dashed):
        self._draw_line(width, color, dashed)

# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
class Label:
    def __init__(self, xml, text, position,
                 fontsize=None, width=None, height=None, color=None):
        x, y = self.position = position
        if not text and text != 0:
            return   # FIX: was `return ""` which is a no-op in __init__
        if not fontsize:
            fontsize = "normal"
        # FIX: str() needed when text is int / float
        text_str = str(text)
        if not width:
            width  = len("m" + text_str) * G.sizes[fontsize]
        if not height:
            height = G.sizes[fontsize] * 2.1
        if not color:
            color  = G.colors['black']
        px, py      = x_to_px(x), y_to_py(y)
        px          = max(-G.pad + 1, px - width / 2)
        py          = py - height / 2
        stroke_size = f"{G.strokesizes[fontsize]}"
        xml.rect(x=px, y=py, width=width, height=height,
                 fill="white", stroke=color, rx=5,
                 **{"stroke-width": stroke_size})
        DrawText(xml, text, position, fontsize=fontsize, width=width)

# ---------------------------------------------------------------------------
# DrawText
# ---------------------------------------------------------------------------
class DrawText:
    def __init__(self, xml, text, position, fontsize="normal", width=None, color=None):
        x, y = self.position = position
        # FIX: was G.colors['dark'] (#555) — too faint on light background;
        #      use 'black' (#000) for proper contrast.
        if color is None:
            color = G.colors['black']
        if not text and text != 0:
            return
        # FIX: str() cast before len()
        text_str = str(text)
        if not width:
            width = len("m" + text_str) * G.sizes[fontsize]
        px, py = x_to_px(x), y_to_py(y)
        px     = max(-G.pad + 1 + width / 2, px)
        py     = py + 1
        font   = f"normal {G.fontsizes[fontsize]}px sans-serif"
        style  = f"dominant-baseline: middle; text-anchor: middle; font: {font}"
        xml.text(text_str, x=px, y=py, width=width, fill=color, style=style)

# ---------------------------------------------------------------------------
# Coordinates helper
# ---------------------------------------------------------------------------
class Coordinates:
    def __init__(self, xml, position, color=None):
        x, y      = self.position = position
        width     = 1
        dashed    = True
        fontsize  = "small"
        Line(xml, from_=[x, 0], to=[x, y], width=width,
             dashed=dashed, color=color)
        Line(xml, from_=[0, y], to=[x, y], width=width,
             dashed=dashed, color=color)
        Label(xml, text=x, position=Position(x, -12 / G.yScale),
              color=color, fontsize=fontsize)
        Label(xml, text=y, position=Position(-12 / G.xScale, y),
              color=color, fontsize=fontsize)

# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------
class Point:
    def __init__(self, xml, position, label=None, color=None, axis=False):
        x, y = self.position = position
        px, py = x_to_px(x), y_to_py(y)
        xml.circle(cx=px, cy=py, r=4, fill=color)
        if label:
            Label(xml, text=label, color=color,
                  position=Position(x, y + 20 / G.yScale))
        if axis:
            Coordinates(xml, position, color=color)

# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------
class Circle:
    def __init__(self, xml, position=Position(0, 0), radius=None,
                 label=None, color=None, width=None):
        x, y = self.position = position
        if not radius: radius = 1
        if not color:  color  = G.colors['black']   # FIX: was bare `colors`
        if not width:  width  = 2
        px, py = x_to_px(x), y_to_py(y)
        xml.ellipse(cx=px, cy=py,
                    rx=radius * G.xScale, ry=radius * G.yScale,
                    fill="none", stroke=color,
                    **{"stroke-width": width, "path-length": "1px"})
        if label:
            Label(xml, text=label, position=position, color=color)
        self.radius = radius
        self.px     = px
        self.py     = py

# ---------------------------------------------------------------------------
# Plot  –  FULLY REIMPLEMENTED
# ---------------------------------------------------------------------------
class Plot:
    """
    Plot a function over the current graph viewport.

    Parameters
    ----------
    xml     : Builder context
    fn      : callable  f(x) -> y   OR  a string expression in x, e.g. "x**2"
    color   : stroke colour (default black)
    width   : stroke width (default 1.75)
    dashed  : dash pattern string, True for default, or False for solid
    n_points: number of sample points (default 200)
    clip    : if True (default), skip line segments outside [y_min, y_max]

    FIX: original used range() with floats (broken) and eval() in a loop (fragile).
         Now uses np.linspace, accepts callables, and clips out-of-range segments.
    """
    def __init__(self, xml, fn, color=None, width=1.75,
                 dashed=False, n_points=200, clip=True):
        if color is None:
            color = G.colors['black']

        # Accept callable or string expression
        if callable(fn):
            func = fn
        else:
            # Compile once for speed; safe-ish for internal use
            code = compile(fn, "<plot>", "eval")
            import math as _math
            func = lambda x, _code=code: eval(_code, {"x": x, "math": _math,
                                                       "__builtins__": {}})

        xs = np.linspace(G.x_min, G.x_max, n_points)
        ys = []
        for x in xs:
            try:
                y = func(float(x))
                ys.append(float(y))
            except Exception:
                ys.append(float('nan'))

        points = list(zip(xs, ys))

        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            # Skip NaN segments
            if math.isnan(y0) or math.isnan(y1):
                continue
            # Clip segments that leave the viewport entirely
            if clip:
                if (max(y0, y1) < G.y_min or min(y0, y1) > G.y_max):
                    continue
            Line(xml,
                 from_ = Position(x0, y0),
                 to    = Position(x1, y1),
                 color = color,
                 width = width,
                 dashed = dashed)

# ---------------------------------------------------------------------------
# Axis / grid drawing helpers
# ---------------------------------------------------------------------------
def draw_axes(xml):
    Arrow(xml, to=[G.x_max, 0])
    Arrow(xml, to=[0, G.y_max])
    DrawText(xml, text=G.axes[0], position=Position(G.x_max, -12 / G.yScale))
    DrawText(xml, text=G.axes[1], position=Position(-12 / G.xScale, G.y_max))


def draw_grid(xml):
    step_size = G.grid_step_size
    color     = G.colors['gray']
    fill      = G.colors['light']
    gridid    = (f"grid-{step_size}-{color}-{fill}"
                 f"-{G.width}-{G.height}-{G.xScale}-{G.yScale}")
    w      = step_size * G.xScale
    h      = step_size * G.yScale
    flipV  = f"transform: scaleY(-1); transform-origin: 0 {G.height / 2}px;"
    with xml.defs:
        with xml.pattern(id="smallGrid", width=w, height=h,
                         patternUnits="userSpaceOnUse"):
            xml.path(d=f"M {w} 0 L 0 0 0 {h}",
                     fill=fill, stroke=color,
                     **{"stroke-width": "0.5"})
        with xml.pattern(id=gridid, width=G.width, height=G.height,
                         patternUnits="userSpaceOnUse"):
            xml.rect(width=G.width, height=G.height,
                     fill="url(#smallGrid)")
            xml.path(d=f"M {G.width} 0 L 0 0 0 {G.height}",
                     fill="none", stroke=color,
                     **{"stroke-width": "0.5"})
    xml.rect(width=G.width, height=G.height,
             fill=f"url(#{gridid})", style=flipV)


def draw_units(xml, color=None):
    def myrange(start, stop, step: float):
        val = start
        while val < stop:
            yield val
            val += step

    if color is None:
        color = G.colors["dark"]
    size = G.grid_step_size

    if size < 0.01:   fixed_places = 3
    elif size < 0.1:  fixed_places = 2
    elif size < 1:    fixed_places = 1
    else:             fixed_places = 0

    # x-axis tick marks + labels
    from_value = size * math.ceil(G.x_min / size)
    for x_pos in myrange(from_value, G.x_max, size):
        Line(xml, from_=[x_pos, 0], to=[x_pos, -5 / G.yScale],
             width=1.5, color=color, dashed=False)
        Label(xml, text=str(round(x_pos, fixed_places)),
              position=Position(x_pos, -14 / G.yScale),
              color=color, fontsize="tiny")

    # y-axis tick marks + labels
    from_value = size * math.ceil(G.y_min / size)
    for y_pos in myrange(from_value, G.y_max, size):
        Line(xml, from_=[0, y_pos], to=[-5 / G.xScale, y_pos],
             width=1.5, color=color, dashed=False)
        Label(xml, text=str(round(y_pos, fixed_places)),
              position=Position(-12 / G.yScale, y_pos),
              color=color, fontsize="tiny", width=20)


def setup_the_axes(xml):
    # 1. solid background rect so inline SVG has a visible background
    xml.rect(
        x      = -G.pad,
        y      = -G.pad * 0.6,
        width  = G.width  + 1.6 * G.pad,
        height = G.height + 1.6 * G.pad,
        fill   = G.colors['light'],
    )
    # 2. grid (drawn before axes so axes sit on top)
    if G.grid_step_size > 0:
        draw_grid(xml)
    # 3. axes and unit labels on top
    if G.axes is not None:
        draw_axes(xml)
        if G.units:
            draw_units(xml)

# ---------------------------------------------------------------------------
# Demo drawing
# ---------------------------------------------------------------------------
def make_drawing(xml):
    if 1:
        Point(xml, position=Position(9, 9), label="point", color="green")
        Point(xml, position=Position(4, 2), label="c",     color="blue")
        Point(xml, position=Position(6, 5), label="d",     color="black", axis=True)
        Point(xml, position=Position(8, 8), label="e",     color="red",   axis=True)
        Line(xml,  label="line",   from_=Position(0, 0), to=Position(4, 8))
        Arrow(xml, label="vector", to=Position(8, 4))
        Label(xml, "NODE", Position(4, 6), color="red")

    if 0:   # vector addition demo
        first  = Arrow(xml, label="a", color="red",  from_=Position(0,0), to=Position(6,8),  axis=False)
        second = Arrow(xml, label="b",               from_=Position(0,0), to=Position(8,4),  axis=True)
        Arrow(xml, label="c", color="blue", from_=first.to, to=second.to, axis=True)

    if 0:   # angle / polygon demo
        Angle(xml, position=Position(0,0), label="First", to=90,   radius=3)
        Angle(xml, position=Position(0,0), label="a",     to=45,   radius=4)
        Angle(xml, position=Position(0,0), label="b",     from_=45, to=90, radius=5, color="red")
        Angle(xml, position=Position(0,0), label="c",     to=45,   radius=5, color="blue", dashed=False)
        Angle(xml, position=Position(7,0), label="d",     radius=2, from_=0, to=180)
        Polygon(xml, points=[(0,0),(1,3),(3,1)], sides="a,b,c")

    if 0:   # function plot demo
        Plot(xml, fn=lambda x: math.sin(x) * 3 + 5, color="blue",   width=2, dashed=False)
        Plot(xml, fn="x**2 / 10",                    color="purple", width=2, dashed=False)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__=="__main__":
    with Builder() as xml:
        with xml.body(style="overflow: hidden; background: #1a1a1a;"):
            with graph(xml, width="20cm", height="20cm", pad=30,
                    x_min=0, x_max=10, y_min=0, y_max=10,
                    dark=False, axes=('x', 'y'),
                    grid_step_size=1, units=False):
                setup_the_axes(xml)
                make_drawing(xml)
            # Inject LeaderLine JavaScript once, after the SVG
            xml.script(LEADER_LINE_JS.strip(), type="text/javascript")
    output_filename = __file__ + ".html"
    with open(output_filename, encoding="utf-8", mode="w") as output_file:
        output_file.write(str(xml))

    # ---------------------------------------------------------------------------
    # Dev server + browser launch
    # ---------------------------------------------------------------------------
    import os
    import threading
    import webbrowser
    from http.server import CGIHTTPRequestHandler, HTTPServer

    port = 8001

    def start_server(path, port=8000):
        os.chdir(path)
        httpd = HTTPServer(('', port), CGIHTTPRequestHandler)
        httpd.serve_forever()

    threading.Thread(
        name   = 'daemon_server',
        target = start_server,
        args   = ('.', port),
        daemon = True,   # FIX: daemon=True so the server dies with the main thread
    ).start()

    webbrowser.open(
        f"http://localhost:{port}/{os.path.basename(output_filename)}",
        new=2, autoraise=True,
    )
