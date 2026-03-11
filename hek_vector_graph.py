#!/usr/bin/env python3
# -*- coding: iso-8859-1 -*-
"""
todo: merge with hek_svg_node_editor.py
todo: add HTMX and layout algorithm
add foreign html objects
<svg width="400" height="400">
  <foreignObject x="10" y="10" width="200" height="100">
    <form>
      <label for="name">Name:</label>
      <input type="text" id="name" name="name"><br><br>
      <label for="email">Email:</label>
      <input type="email" id="email" name="email"><br><br>
      <input type="submit" value="Submit">
    </form>
  </foreignObject>
</svg>

or
<svg xmlns = "http://www.w3.org/2000/svg" width="100%" height="100%">
    <rect x="25" y="25" width="250" height="200" fill="#ff0000" stroke="#000000"/>
    <foreignObject x="50" y="50" width="200" height="150">
        <body xmlns="http://www.w3.org/1999/xhtml">
            <form>
                <input type="text"/>
                <input type="text"/>
            </form>
        </body>
    </foreignObject>
    <circle cx="60" cy="80" r="30" fill="#00ff00" fill-opacity="0.5"/>
</svg>

or

<svg width="100%" height="500" xmlns="http://www.w3.org/2000/svg>
  <foreignObject x="10" y="10" width="100" height="150">
      <div xmlns="http://www.w3.org/1999/xhtml">
      <input></input>
          </div>
  </foreignObject>
</svg>
"""
import numpy as np
import math, re
from py2html import Builder, SELFCLOSING
from hek_map_utils import Vector, Position
##################################################################################################################
# convert to pixel coordinates: from (x,y) to (px, py)
x_to_px=lambda x: (x-G.x_min)*G.xScale
y_to_py=lambda y: G.height - (y - G.y_min) * G.yScale
# conversion factor between cm and pixels
CM=37.79527559
##################################################################################################################
class G:
    sizes = {'tiny': 8, 'small': 8, 'normal': 10, 'large': 12}
    fontsizes = {'tiny': 12, 'small': 12, 'normal': 16, 'large': 20}
    strokesizes = {'tiny': 0, 'small': 1, 'normal': 1.75, 'large': 2}
    global_id=0
def graph(xml, width=200, height=200, pad=30, x_min=0, x_max=10, y_min=0, y_max=10, grid_step_size=1, dark=False, axes=("x","y"), units=True, id="PLACEHOLDER"):
    if "cm" in width:
        width=int(width[:-2])*CM
    G.width=width
    if "cm" in height:
        height=int(height[:-2])*CM
    G.height=height
    G.x_min=x_min
    G.x_max=x_max
    G.y_min=y_min
    G.y_max=y_max
    G.grid_step_size=grid_step_size
    G.axes=axes
    G.units=units
    G.xScale = width / (x_max - x_min)
    G.yScale = height / (y_max - y_min)
    G.colors = {
        'light': 'black' if dark else 'white',
        'gray': 'gray' if dark else '#ccc',
        'dark': 'dark' if dark else '#aaa',
        'black': 'white' if dark else '#000'
    }
    G.pad=pad
    return xml.svg( id=id,
            width=f"{width}",
            height=f"{height}",
            viewbox=f"{-pad} {-pad * 0.6} {width + 1.6 * pad} {height + 1.6 * pad}",
            xmlns="http://www.w3.org/2000/svg",
            style=f"background: {G.colors['light']}; border-radius: 8px;")

class LeaderLine:
    def __init__(self, xml, start_node, end_node, start_slot, end_slot, **options):
        xml.line(visibility="hidden", class_="connection", start_node=start_node, end_node=end_node, start_slot=start_slot, end_slot=end_slot, **options)

def ForeignObject(xml, px, py, width, height):
    "width and height are given in pixels !!!!"
    return xml.foreignObject(x=px, y=py, width=width, height=height)
if 0:
    with ForeignObject(xml,Position(x_to_px(10),y_to_py(10)),200,200):
        with xml.form:
            xml.label("Name:", for_="name"); xml.input(type="text", id="name", name="name")
            xml.br(SELFCLOSING)
            xml.label("Email:",for_="email"); xml.input(type="email",id="email",name="email")
            xml.br(SELFCLOSING)
            xml.input(type="submit",value="Submit")

class Angle:
    def __init__(self, xml, position, label, from_=0, to=0, radius=None, color=None, fontsize=None, dashed=None):
        x,y= self.position=position
        def draw_arc(x, y, r, from_, to):
            def to_euclidian(x, y, radius, rads):
                return [int(x + radius * math.cos(rads)), int(y + radius * math.sin(rads))]
            from_rads = (-from_ * math.pi) / 180
            to_rads = (-to * math.pi) / 180

            large = "0" if to - from_ <= 180 else "1"
            x_start, y_start = to_euclidian(x, y, r, from_rads)
            x_end, y_end = to_euclidian(x, y, r, to_rads)
            return f"M {x_start} {y_start} A {int(r)} {int(r)} 0 {large} 0 {x_end} {y_end}"
        if radius is None:
            radius = (G.x_max - G.x_min) / 3
        self.x=x
        self.y=y
        self.fr=from_
        self.to=to
        self.radius=radius
        if color is None:
            color = G.colors['black']
        if fontsize is None:
            fontsize = "small"
        if dashed is None:
            dashed = "5,3"
        # Convert angles to radians
        from_angle = np.deg2rad(from_)
        to_angle = np.deg2rad(to)

        # Check if angle difference is greater than 180 degrees
        if abs(to_angle - from_angle) > np.pi:
            label_angle = (from_angle + to_angle) + np.pi
        else:
            label_angle = (from_angle + to_angle) / 2

        if 0: # buggy
            # Adjust to and from to take into account the scales
            tan_to = math.tan((to_angle * math.pi) / 180)
            to_angle = (math.atan2(tan_to * G.yScale, G.xScale) * 180) / math.pi

            tan_from = math.tan((from_angle * math.pi) / 180)
            from_angle = (math.atan2(tan_from * G.yScale, G.xScale) * 180) / math.pi
        else:
            to_angle = np.rad2deg(to_angle)
            from_angle = np.rad2deg(from_angle)
        # Adjust angles for angle range
        if from_angle >= 180:
            from_angle -= 360
        if to_angle >= 180:
            to_angle -= 360

        # Round angles to nearest integer
        from_angle = int(from_angle) % 360
        to_angle = int(to_angle) % 360
        px,py= x_to_px(x), y_to_py(y)

        xml.path(fill="none",
                stroke=color,
                d=draw_arc(px, py, radius * G.xScale, from_angle, to_angle),
                **{"stroke-dasharray":dashed,"stroke-width":"1"}
                )

        # Calculate label position (adjust for G.xScale vs G.yScale)
        if G.xScale == G.yScale:
            x_offset = radius * np.cos(label_angle)
            y_offset = radius * np.sin(label_angle)
        else:
            # Adjust for aspect ratio differences
            x_offset = radius * np.cos(label_angle) * G.xScale / G.yScale
            y_offset = radius * np.sin(label_angle)
        #label="\u03B1"
        #print(label)
        Label(xml, text= label, position=Position(x+x_offset,y+y_offset), fontsize=fontsize, color="blue")

class Polygon:
    def __init__(self, xml, points, sides=[], angles=[], color=None):
        self.points=points
        if color is None:
            color=G.colors["black"]
        sides = sides.split(",")

        for i in range(len(points)):
            from_point = points[i]
            to_point = points[(i + 1) % len(points)]
            label = sides[i] if sides else None
            Line(xml, from_=from_point, to=to_point, label=label, color=color)
        if angles:
            #print(angles)
            if isinstance(angles, str):
                angles = angles.split(",")
            #print('angles =', ', '.join(angles))
            for i in range(len(points)):
                from_point = points[i]
                next_point = points[(i + 1) % len(points)]
                prev_point = points[(len(points) + i - 1) % len(points)]
                size = math.sqrt(
                    (from_point[0] - next_point[0]) ** 2 + (from_point[1] - next_point[1]) ** 2
                )
                next_vec = [next_point[0] - from_point[0], next_point[1] - from_point[1]]
                prev_vec = [from_point[0] - prev_point[0], from_point[1] - prev_point[1]]

                prev_angle = round(
                    180 + (math.atan2(prev_vec[1], prev_vec[0]) * 180) / math.pi
                )
                next_angle = round(
                    (360 + (math.atan2(next_vec[1], next_vec[0]) * 180) / math.pi) % 360
                )
                radius = size / 3
                Angle(xml,
                            position= from_point,
                            label= angles[i],
                            from_= prev_angle,
                            to= next_angle,
                            color= color
                            )

class Line:
    def __init__(self, xml, to, from_=Position(0,0), label=None, width=1.75, dashed=True, color=None, axis=False):
        self.to=to
        self.fr=from_
        if color is None:
            color = G.colors['black']
        if dashed:
            dashed="5,3"
        self.draw_line(width, color, dashed)
        if label:
            # Calculate label position (midpoint)
            label_x = (to[0] + from_[0]) / 2
            label_y = (to[1] + from_[1]) / 2
            Label(xml, text=label, color=color, position=Position(label_x,label_y))
        if axis:
            Coordinates(xml, position=to, color=color)

    def draw_line(self,width, color, dashed):
        # Convert line coordinates to pixel values
        px1,py1= x_to_px(self.fr[0]), y_to_py(self.fr[1])
        px2,py2= x_to_px(self.to[0]), y_to_py(self.to[1])
        xml.line(**{"x1":px1, "y1":py1, "x2":px2, "y2":py2, "stroke":color, "stroke-width":width, "stroke-dasharray":dashed})

class Arrow(Line):
    def draw_line(self,width, color, dashed):
        # Convert vector coordinates to pixel values
        px1,py1= x_to_px(self.fr[0]), y_to_py(self.fr[1])
        # Calculate vector magnitude and angle
        to_pix = [(self.to[0] - self.fr[0]) * G.xScale, (self.to[1] - self.fr[1]) * G.yScale]
        mag = np.sqrt(to_pix[0]**2 + to_pix[1]**2)
        angle = np.arctan2(to_pix[1], to_pix[0])

        # Adjust vector length to account for arrow head
        arr_size = 10 # arrow head size
        new_mag = mag - 2 * arr_size

        # Calculate end point of the vector
        px2 = px1 + new_mag * np.cos(angle)
        py2 = py1 - new_mag * np.sin(angle)

        # Generate a unique identifier for the vector element
        global_id = G.global_id
        G.global_id += 1
        with xml.defs:
            with xml.marker(id=f"h-{global_id}", markerWidth="10", markerHeight="5", refY="2.5", orient="auto"):
               xml.polygon(points="0 0, 10 2.5, 0 5", fill=color)
        xml.line(x1=px1, y1=py1, x2=px2, y2=py2, stroke=color,**{"stroke-width":width, "marker-end":f"url(#h-{global_id})"})

class Label:
    def __init__(self, xml, text, position, fontsize=None, width=None, height=None, color=None):
        x,y= self.position= position
        # CAVEAT width and height are in pixels !!!!
        if not text:
            return ""
        if not fontsize:
            fontsize = "normal"
        if not width:
            width = len("m" + str(text)) * G.sizes[fontsize]
        if not height:
            height = G.sizes[fontsize] * 2.1
        if not color:
            color = G.colors['black']
        px,py=x_to_px(x), y_to_py(y)
        px = max(-G.pad + 1, px-width/2)
        py = py-height/2
        stroke_size=f"{G.strokesizes[fontsize]}"
        xml.rect(x=px, y=py, width=width, height=height,
              fill="#fffd",
              stroke=color,
              rx=5,
              **{"stroke-width":stroke_size})
        DrawText(xml,text, position, fontsize=fontsize, width=width)

class DrawText:
    def __init__(self, xml, text, position, fontsize="normal", width=None):
        x,y= self.position=position
        color = G.colors['dark']
        if not text:
            return ""
        if not width:
            width = len("m" + text) * G.sizes[fontsize]
        px, py=x_to_px(x), y_to_py(y)
        px = max(-G.pad+1+ width/2, px)
        py = py + 1

        font = f"normal {G.fontsizes[fontsize]}px sans-serif"
        style = f"dominant-baseline: middle; text-anchor: middle; font: {font}"
        xml.text(text,x=px, y=py, width=width, fill=color, style=style)

class Coordinates:
    def __init__(self, xml, position, color=None):
        x,y= self.position=position
        width = 1
        dashed = True
        fontsize = "small"
        Line(xml, from_=[x, 0], to= [x, y], width=width, dashed=dashed, color=color)
        Line(xml, from_=[0, y], to= [x, y], width=width, dashed=dashed, color=color)
        Label(xml, text=x, position=Position(x, -12/G.yScale), color=color, fontsize=fontsize)
        Label(xml, text=y, position=Position(-12/G.xScale,y), color=color, fontsize=fontsize)

class Point:
    def __init__(self, xml, position, label=None, color=None, axis=False):
        x,y= self.position=position
        px,py= x_to_px(x), y_to_py(y)
        xml.circle(cx=px, cy=py, r=4, fill=color)
        if label:
            Label(xml, text=label, color=color, position=Position(x,y+20/G.yScale))
        if axis:
            Coordinates(xml, position, color=color)

class Circle:
    def __init__(self, xml, position=Position(0,0), radius=None, label=None, color=None, width=None):
        x,y= self.position=position
        if not radius:
            radius = 1
        if not color:
            color = colors['black']
        if not width:
            width = 2
        px,py= x_to_px(x), y_to_py(y)
        xml.ellipse(cx=px, cy=py, rx=radius * G.xScale, ry=radius * G.yScale,
            fill="none", stroke=color, **{"stroke-width":width,"path-length":"1px"})
        if label:
            Label(xml, text=label, position=position, color=color)
        self.radius=radius
        self.px=px
        self.py=py

class Plot:
    def __init__(self, xml, fn, color, width, dashed):
        points = []
        xUnit = G.x_max / G.width
        for x in range(G.x_min, G.width, xUnit):
            y = eval(fn.replace("x", str(x)))
            points.append([x, y])
        for i, p in enumerate(points[:-1]):
            Line(xml,from_=p, to=points[i + 1], color=color, width=width, dashed=dashed)
###############################################################################################################
def draw_axes(xml):
    color = G.colors['dark']
    Arrow(xml, to= [G.x_max, 0]) # the x-axis
    Arrow(xml, to= [0, G.y_max]) # the y-axis
    DrawText(xml, text= G.axes[0], position=Position(G.x_max, -12/G.yScale))
    DrawText(xml, text= G.axes[1], position=Position(-12/G.xScale, G.y_max))

def draw_grid(xml):
    step_size = G.grid_step_size
    color = G.colors['gray']
    fill = G.colors['light']
    gridid = f"grid-{step_size}-{color}-{fill}-{G.width}-{G.height}-{G.xScale}-{G.yScale}"
    w = step_size * G.xScale
    h = step_size * G.yScale
    big = max(h,w)
    flipV = f"transform: scaleY(-1); transform-origin: 0 {G.height/2}px;"
    with xml.defs:
        with xml.pattern(id="smallGrid", width=w, height=h, patternUnits="userSpaceOnUse"):
            xml.path(d=f"M {w} 0 L 0 0 0 {h}",
                fill=fill,
                stroke=color,
                **{"stroke-width":"0.5"})
        with xml.pattern(id=gridid, width=G.width, height=G.height, patternUnits="userSpaceOnUse"):
            xml.rect(width=G.width,height=G.height, fill="url(#smallGrid)")
            xml.path(d=f"M {G.width} 0 L 0 0 0 {G.height}",
                    fill="none",
                    stroke=color,
                    **{"stroke-width":"0.5"})
    xml.rect(width=G.width,height=G.height,fill=f"url(#{gridid})", style=flipV)

def draw_units(xml, color=None):
    def Myrange(start,stop,step: float): # range but for floats
        val=start
        while True:
            yield val
            val+=step
            if val>= stop:
                break
    if color is None:
        color = G.colors["dark"]
    size=G.grid_step_size
    # Initialize fixed decimal places based on unit size
    if size < 0.01:
        fixed_places = 3
    elif size < 0.1:
        fixed_places = 2
    elif size < 1:
        fixed_places = 1
    else:
        fixed_places = 0

    # x-axis
    # Round starting value to nearest unit multiple
    from_value = int(size * math.ceil(G.x_min / size))
    to_value=int(G.x_max)
    # Iterate through units and draw lines and labels
    for x_pos in Myrange(from_value, to_value, size):
        Line(xml, from_=[x_pos, 0], to=[x_pos, -5 / G.yScale], width= 1.5, color=color)
        Label(xml, text= str(round(x_pos, fixed_places)),
                position=Position(x_pos,-14 / G.yScale),
                color=color,
                fontsize="tiny")

    # y-axis
    # Round starting value to nearest unit multiple
    from_value = int(size * math.ceil(G.y_min / size))
    to_value=int(G.y_max)
    # Iterate through units and draw lines and labels
    for y_pos in Myrange(from_value, to_value, size):
        Line(xml, from_=[0, y_pos], to=[-5 / G.xScale, y_pos], width= 1.5, color=color)
        Label(xml, text= str(round(y_pos, fixed_places)),
                position=Position(-12 / G.yScale,y_pos),
                color=color,
                fontsize="tiny",
                width=20)

def setup_the_axes(xml):
    if G.axes is not None: #
        draw_axes(xml)
        if G.units:
            draw_units(xml)
    if G.grid_step_size>0:
        draw_grid(xml)

def make_drawing(xml):
    if 1:
        Point(xml, position=Position(9,9), label="point", color="green")
        Point(xml, position=Position(4,2), label="c", color="blue")
        Point(xml, position=Position(6,5), label="d", color="black",axis=True)
        Point(xml, position=Position(8,8), label="e", color="red", axis=True)
        Line(xml,label="line", from_=Position(0,0), to=Position(4,8))
        Arrow(xml, label="vector", to=Position(8,4))
        Label(xml, "NODE", Position(4,6), color="red")
    if 0:
        first=Arrow(xml, label="a", color="red", from_=Position(0,0), to=Position(6,8), axis=False)
        second=Arrow(xml, label="b", from_=Position(0,0), to=Position(8,4), axis=True)
        Arrow(xml, label="c", color="blue", from_=first.to, to=second.to, axis=True)
    if 0:
        Angle(xml,position=Position(0,0), label="First",to=90, radius=3)
        Angle(xml,position=Position(0,0), label="a", to=45, radius=4)
        Angle(xml,position=Position(0,0), label="b", from_=45,to=90,radius=5, color="red")
        Angle(xml,position=Position(0,0), label="c", to=45,radius=5, color="blue", dashed=False)
        Angle(xml,position=Position(7,0), label="d", radius=2, from_=0, to=180)
        Polygon(xml,points=[(0,0),(1,3),(3,1)], sides="a,b,c")

with Builder() as xml:
    if 0:
        with graph(xml, id="TRIANGLE", width="30cm", height="25cm", pad=30, x_min=0, x_max=3, y_min=0, y_max=3, dark=False,
                                                                                axes=None, grid_step_size=1, units=False):
            Polygon(xml,points=[(0,0),(1,3),(3,1)], sides="a,b,c", angles=['α','β','γ']) # fix the unicode problen
            with ForeignObject(xml,Position(0,3),200,200): # y-coordinates are inverted !!!
                with xml.form:
                    xml.label("Name:", for_="name"); xml.input(type="text", id="name", name="name")
                    xml.br(SELFCLOSING)
                    xml.label("Email:",for_="email"); xml.input(type="email",id="email",name="email")
                    xml.br(SELFCLOSING)
                    xml.input(type="submit",value="Submit")
    else:
        #with graph(xml):
        #with graph(xml, width=200, height=200, pad=30, x_min=0, x_max=10, y_min=0, y_max=10, dark=False, axes=None, grid_step_size=1, units=True):
        #with graph(xml, width=200, height=200, pad=30, x_min=0, x_max=1, y_min=0, y_max=1, dark=False, axes=('x','y'), grid_step_size=0.2, units=True):
        with xml.body(style="overflow: hidden; background: #1a1a1a;"):
            with graph(xml, width="20cm", height="20cm", pad=30, x_min=0, x_max=10, y_min=0, y_max=10, dark=False,
                                                                axes=('x','y'), grid_step_size=1, units=False):
                setup_the_axes(xml)
                make_drawing(xml)

output_filename=__file__+".html"
#with open(output_filename, encoding="iso-8859-1", mode="w") as output_file:
with open(output_filename, encoding="utf-8", mode="w") as output_file:
    output_file.write(str(xml))
from http.server import HTTPServer, CGIHTTPRequestHandler
import os, webbrowser, threading
# Start the server in a new thread, so that we can open the browser in the main thread
port = 8001
def start_server(path, port=8000):
    os.chdir(path)
    httpd = HTTPServer(('', port), CGIHTTPRequestHandler)
    httpd.serve_forever()
threading.Thread(name='daemon_server',
                  target=start_server,
                  args=('.', port),
                  daemon=False # Set daemon to false so it will be killed once the main thread is dead.
                  ).start()
webbrowser.open(
    f"http://localhost:{port}/{os.path.basename(output_filename)}", new=2, autoraise=True)
