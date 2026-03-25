"""
hek_map_utils.py — Geospatial geometry primitives
==================================================
Provides three layers of abstraction:

  1. Coordinate conversion helpers
       dms2dec / dd2dms / human_latlong2dec / dec2human_latlong
       Convert between human-readable DMS strings and decimal degrees.

  2. Flat-plane geometry  (Vector, Position)
       2-D vector arithmetic built on top of Python's complex numbers.
       Distances are in metres; angles are in decimal degrees.

  3. Geodetic / great-circle geometry  (GeoVector, GeoPoint)
       Uses the Vincenty direct/inverse formulae (pure-Python) so that
       arithmetic on geographic coordinates (lat/long in decimal degrees)
       works with the same operator syntax as the flat-plane classes.
       Distances are in kilometres; bearings are clockwise from true north.

Dependencies: none (stdlib only)
"""

import cmath
from math import (
    atan, atan2, cos, degrees, fabs, radians, sin, sqrt, tan, pi, asin,
)
from rich.traceback import install; install()


# ---------------------------------------------------------------------------
# WGS-84 ellipsoid parameters
# ---------------------------------------------------------------------------

_WGS84_A = 6_378_137.0          # semi-major axis (m)
_WGS84_F = 1 / 298.257223563    # flattening
_WGS84_B = _WGS84_A * (1 - _WGS84_F)  # semi-minor axis (m)


# ---------------------------------------------------------------------------
# Vincenty direct formula
# ---------------------------------------------------------------------------

def _vincdir(lat1_deg: float, lon1_deg: float, az1_deg: float, dist_m: float):
    """Vincenty direct: given a start point, azimuth, and distance, return the
    end point and reverse azimuth.

    Returns:
        (lat2_deg, lon2_deg, az2_deg)  — all in decimal degrees.
    """
    a, b, f = _WGS84_A, _WGS84_B, _WGS84_F
    lat1 = radians(lat1_deg)
    lon1 = radians(lon1_deg)
    az1  = radians(az1_deg)

    tan_U1 = (1 - f) * tan(lat1)
    cos_U1 = 1 / sqrt(1 + tan_U1**2)
    sin_U1 = tan_U1 * cos_U1

    sin_az1, cos_az1 = sin(az1), cos(az1)
    sigma1 = atan2(tan_U1, cos_az1)
    sin_alpha = cos_U1 * sin_az1
    cos2_alpha = 1 - sin_alpha**2
    u2 = cos2_alpha * (a**2 - b**2) / b**2
    A_ = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B_ = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    sigma = dist_m / (b * A_)
    for _ in range(100):
        cos2sm = cos(2 * sigma1 + sigma)
        sin_sigma = sin(sigma)
        cos_sigma = cos(sigma)
        d_sigma = B_ * sin_sigma * (
            cos2sm + B_ / 4 * (
                cos_sigma * (-1 + 2 * cos2sm**2)
                - B_ / 6 * cos2sm * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos2sm**2)
            )
        )
        sigma_new = dist_m / (b * A_) + d_sigma
        if fabs(sigma_new - sigma) < 1e-12:
            sigma = sigma_new
            break
        sigma = sigma_new

    cos2sm = cos(2 * sigma1 + sigma)
    sin_sigma, cos_sigma = sin(sigma), cos(sigma)

    lat2 = atan2(
        sin_U1 * cos_sigma + cos_U1 * sin_sigma * cos_az1,
        (1 - f) * sqrt(sin_alpha**2 + (sin_U1 * sin_sigma - cos_U1 * cos_sigma * cos_az1)**2),
    )
    lam = atan2(
        sin_sigma * sin_az1,
        cos_U1 * cos_sigma - sin_U1 * sin_sigma * cos_az1,
    )
    C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
    L = lam - (1 - C) * f * sin_alpha * (
        sigma + C * sin_sigma * (cos2sm + C * cos_sigma * (-1 + 2 * cos2sm**2))
    )
    lon2 = lon1 + L
    az2 = atan2(sin_alpha, -sin_U1 * sin_sigma + cos_U1 * cos_sigma * cos_az1)

    return degrees(lat2), degrees(lon2), (degrees(az2) + 360) % 360


# ---------------------------------------------------------------------------
# Vincenty inverse formula
# ---------------------------------------------------------------------------

def _vincinv(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float):
    """Vincenty inverse: given two points, return the geodetic distance and
    forward/reverse azimuths.

    Returns:
        (dist_m, az1_deg, az2_deg)
    """
    a, b, f = _WGS84_A, _WGS84_B, _WGS84_F

    if fabs(lat1_deg - lat2_deg) < 1e-10 and fabs(lon1_deg - lon2_deg) < 1e-10:
        return 0.0, 0.0, 0.0  # coincident points

    u1 = atan((1 - f) * tan(radians(lat1_deg)))
    u2 = atan((1 - f) * tan(radians(lat2_deg)))

    lon = radians(lon2_deg - lon1_deg)
    omega = lon

    alpha = sigma = cos2sm = 0.0
    for _ in range(1000):
        sin_sigma = sqrt(
            (cos(u2) * sin(lon))**2
            + (cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lon))**2
        )
        cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lon)
        sigma = atan2(sin_sigma, cos_sigma)
        # asin clamps to principal value, preventing sign-flip oscillation
        # near antipodal equatorial points
        alpha = asin((cos(u1) * cos(u2) * sin(lon)) / sin_sigma)
        cos2sm = cos(sigma) - 2 * sin(u1) * sin(u2) / cos(alpha)**2
        c = f / 16 * cos(alpha)**2 * (4 + f * (4 - 3 * cos(alpha)**2))
        new_lon = omega + (1 - c) * f * sin(alpha) * (
            sigma + c * sin(sigma) * (
                cos2sm + c * cos(sigma) * (-1 + 2 * cos2sm**2)
            )
        )
        if fabs(new_lon - lon) < 1e-12:
            lon = new_lon
            break
        lon = new_lon

    u2_sq = cos(alpha)**2 * (a**2 - b**2) / b**2
    A_ = 1 + u2_sq / 16384 * (4096 + u2_sq * (-768 + u2_sq * (320 - 175 * u2_sq)))
    B_ = u2_sq / 1024 * (256 + u2_sq * (-128 + u2_sq * (74 - 47 * u2_sq)))
    d_sigma = B_ * sin(sigma) * (
        cos2sm + B_ / 4 * (
            cos(sigma) * (-1 + 2 * cos2sm**2)
            - B_ / 6 * cos2sm * (-3 + 4 * sin(sigma)**2) * (-3 + 4 * cos2sm**2)
        )
    )
    dist_m = b * A_ * (sigma - d_sigma)

    az1 = degrees(atan2(cos(u2) * sin(lon), cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lon)))
    if az1 < 0:
        az1 += 360
    az2 = degrees(atan2(cos(u1) * sin(lon), -sin(u1) * cos(u2) + cos(u1) * sin(u2) * cos(lon))) + 180

    return dist_m, az1, az2


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def dms2dec(pos_in_dms: str, is_longitude: bool = False) -> float:
    """Convert a DMS string to decimal degrees.

    The direction letter (N/S/E/W) must be the *last* character.
    Latitude strings have 2-digit degree fields; longitude strings have 3.

    Examples::

        dms2dec('043304N')               ->  43.551111...
        dms2dec('0395649E', is_longitude=True) ->  39.947000...
    """
    pos, direction = pos_in_dms[:-1], pos_in_dms[-1]
    if is_longitude:  # longitude: degrees 0–180, 3 digits
        degrees_part = pos[:3]
        minutes_part = pos[3:5]
        seconds_part = pos[5:]
    else:  # latitude: degrees 0–90, 2 digits
        degrees_part = pos[:2]
        minutes_part = pos[2:4]
        seconds_part = pos[4:]

    # Normalise seconds: insert a decimal point, guaranteeing at least one digit
    # on each side so float() never receives a bare '.'.
    if "." not in seconds_part:
        int_s = seconds_part[:2] if len(seconds_part) >= 2 else seconds_part.zfill(2)
        frac_s = seconds_part[2:] if len(seconds_part) > 2 else "0"
        seconds_part = int_s + "." + frac_s

    dd = int(degrees_part) + int(minutes_part) / 60 + float(seconds_part) / 3600
    return -dd if direction in "SW-" else dd


def dd2dms(deg: float, dir: str = "NS") -> str:
    """Convert decimal degrees to a DMS string with a cardinal-direction suffix.

    Args:
        deg: Decimal degrees (positive = N or E, negative = S or W).
        dir: ``'NS'`` for latitude, ``'EW'`` for longitude.

    Returns:
        A zero-padded DMS string, e.g. ``'505405N'`` or ``'0042904E'``.
    """
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = int(round((md - m) * 60))

    # Carry over when rounding pushes seconds to 60
    if sd == 60:
        sd = 0
        m += 1
        if m == 60:
            m = 0
            d += 1

    suffix = dir[0] if deg >= 0 else dir[1]
    if dir == "NS":
        return f"{abs(d):02d}{m:02d}{sd:02d}{suffix}"
    else:
        return f"{abs(d):03d}{m:02d}{sd:02d}{suffix}"


def human_latlong2dec(latlong: str) -> tuple[float, float]:
    """Parse a space-separated ``'DDMMSSN DDDMMSSE'`` string into (lat, long) decimal degrees."""
    lat_str, long_str = latlong.split()
    return dms2dec(lat_str), dms2dec(long_str, is_longitude=True)


def dec2human_latlong(lat_dec: float, long_dec: float) -> str:
    """Format decimal (lat, long) as a human-readable DMS string."""
    return f'{dd2dms(lat_dec, "NS")} {dd2dms(long_dec, "EW")}'


# ---------------------------------------------------------------------------
# Flat-plane geometry — Vector
# ---------------------------------------------------------------------------


class Vector(complex):
    """A 2-D vector stored internally as a complex number.

    Args:
        radius_m:         Length of the vector in metres.
        angle_in_degree:  Angle in decimal degrees (counter-clockwise from the
                          positive-x axis, following the standard math convention).
    """

    def __new__(cls, radius_m: float, angle_in_degree: float) -> Vector:
        z = cmath.rect(radius_m, radians(angle_in_degree))
        return complex.__new__(cls, z.real, z.imag)

    # -- components ----------------------------------------------------------

    @property
    def x(self) -> float:
        """Horizontal component in metres."""
        return self.real

    @property
    def y(self) -> float:
        """Vertical component in metres."""
        return self.imag

    @property
    def length(self) -> float:
        """Magnitude of the vector in metres."""
        return abs(self)

    @property
    def angle(self) -> float:
        """Direction of the vector in decimal degrees."""
        return degrees(cmath.phase(self))

    # -- transformations -----------------------------------------------------

    def rotated_by(self, relative_angle_in_degree: float) -> Vector:
        """Return a new vector rotated counter-clockwise by the given angle."""
        return Vector(self.length, self.angle + relative_angle_in_degree)

    def shortened_by(self, delta: float) -> Vector:
        """Return a new vector with its length reduced by *delta* metres."""
        return Vector(self.length - delta, self.angle)

    def unit(self) -> Vector:
        """Return the unit vector (length 1) in the same direction."""
        return self / abs(self)

    def normal(self) -> Vector:
        """Return the unit normal (90° counter-clockwise rotation)."""
        return self.unit() * 1j

    # -- products / projections ----------------------------------------------

    def dot_product(self, other: Vector) -> float:
        """Scalar dot product."""
        return (self.conjugate() * other).real

    def cross(self, other: Vector) -> float:
        """Z-component of the 3-D cross product (signed area of the parallelogram)."""
        return (self.conjugate() * other).imag

    def cos_theta(self, other: Vector) -> float:
        """Cosine of the angle between *self* and *other*."""
        return self.dot_product(other) / (self.length * other.length)

    def projection(self, onto: Vector) -> Vector:
        """Project *self* onto *onto*."""
        u = onto.unit()
        return self.dot_product(u) * u

    def triangle_area(self, other: Vector) -> float:
        """Signed area of the triangle formed by *self* and *other*."""
        return self.cross(other) / 2

    # -- arithmetic operators ------------------------------------------------

    def __add__(self, other: Vector) -> Vector:
        z = complex(self.x + other.x, self.y + other.y)
        return Vector(abs(z), degrees(cmath.phase(z)))

    def __sub__(self, other: Vector) -> Vector:
        z = complex(self.x - other.x, self.y - other.y)
        return Vector(abs(z), degrees(cmath.phase(z)))

    def __mul__(self, scale: float) -> Vector:
        return Vector(self.length * scale, self.angle)

    def __truediv__(self, scale: float) -> Vector:
        return Vector(self.length / scale, self.angle)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Vector(length={self.length:.2f} m, angle={self.angle:.2f}°)"

    def __str__(self) -> str:
        return f"{self.length:.2f} m, {self.angle:.2f}°"

    def __format__(self, format_spec: str) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Flat-plane geometry — Position
# ---------------------------------------------------------------------------


class Position(tuple):
    """A point in a 2-D orthonormal coordinate system.

    Adding a :class:`Vector` to a :class:`Position` yields a new
    :class:`Position`; subtracting two :class:`Position` objects yields the
    :class:`Vector` between them.
    """

    def __new__(cls, x: float, y: float) -> Position:
        return tuple.__new__(cls, (x, y))

    @property
    def x(self) -> float:
        return self[0]

    @property
    def y(self) -> float:
        return self[1]

    def __add__(self, other: Vector) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Position) -> Vector:
        """Return the displacement vector from *other* to *self*."""
        z = complex(self.x - other.x, self.y - other.y)
        return Vector(abs(z), degrees(cmath.phase(z)))

    def rotate_around_point(self, pivot: Position, angle_in_degree: float) -> Position:
        """Return this position rotated counter-clockwise around *pivot*."""
        vec: Vector = self - pivot
        vec = vec.rotated_by(angle_in_degree)
        return pivot + vec

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"


# ---------------------------------------------------------------------------
# Geodetic geometry — GeoVector
# ---------------------------------------------------------------------------


class GeoVector(complex):
    """A displacement vector on the Earth's surface.

    Internally stored in metres and radians (complex number).
    The external interface uses **kilometres** and **decimal degrees**.

    Bearing convention (clockwise from true north):
        0°  → north,  90° → east,  180° → south,  −90° → west.

    Construction — three equivalent forms::

        GeoVector(100, 45)                              # positional
        GeoVector(distance_km=100, bearing_deg=45)      # named
        GeoVector.from_components(east_km=70, north_km=70)  # Cartesian
    """

    def __new__(cls, distance_km: float, bearing_deg: float) -> GeoVector:
        """Create a GeoVector from a distance and a bearing.

        Args:
            distance_km: Length of the displacement in kilometres.
            bearing_deg: Direction in decimal degrees, clockwise from true north.
        """
        z = cmath.rect(distance_km * 1000, radians(bearing_deg))
        return complex.__new__(cls, z.real, z.imag)

    @classmethod
    def from_components(cls, *, east_km: float, north_km: float) -> GeoVector:
        """Create a GeoVector from orthogonal east/north components.

        Both arguments are keyword-only to prevent east/north confusion.

        Args:
            east_km:  Eastward displacement in kilometres (negative = west).
            north_km: Northward displacement in kilometres (negative = south).

        Example::

            GeoVector.from_components(east_km=-100, north_km=0)  # 100 km due west
        """
        # Internal encoding: north maps to the real axis, east to the imaginary axis,
        # because cmath.rect(dist, radians(bearing)) places bearing=0 (north) on the
        # real axis and bearing=90 (east) on the imaginary axis.
        z = complex(north_km * 1000, east_km * 1000)
        distance_km = abs(z) / 1000
        bearing_deg = degrees(cmath.phase(z)) % 360
        return cls(distance_km=distance_km, bearing_deg=bearing_deg)

    @property
    def magnitude(self) -> float:
        """Length of the vector in kilometres."""
        return abs(self) / 1000

    @property
    def phase(self) -> float:
        """Bearing in decimal degrees (clockwise from north), normalised to [0, 360)."""
        return degrees(cmath.phase(self)) % 360

    @property
    def east_km(self) -> float:
        """Eastward component in kilometres."""
        return self.imag / 1000  # east → imaginary axis in internal encoding

    @property
    def north_km(self) -> float:
        """Northward component in kilometres."""
        return self.real / 1000  # north → real axis in internal encoding

    def turn(self, angle_in_degree: float) -> GeoVector:
        """Return a new vector rotated by *angle_in_degree* degrees."""
        return GeoVector(
            distance_km=self.magnitude, bearing_deg=self.phase + angle_in_degree
        )

    def dot_product(self, other: GeoVector) -> float:
        """Scalar dot product (result in m²)."""
        return (self.conjugate() * other).real

    def cross_product(self, other: GeoVector) -> float:
        """Z-component of the 3-D cross product (result in m²)."""
        return (self.conjugate() * other).imag

    def __truediv__(self, scale: float) -> GeoVector:
        return GeoVector(distance_km=self.magnitude / scale, bearing_deg=self.phase)

    def __mul__(self, scale: float) -> GeoVector:
        return GeoVector(distance_km=self.magnitude * scale, bearing_deg=self.phase)

    def __repr__(self) -> str:
        return (
            f"GeoVector(distance_km={self.magnitude:.2f}, bearing_deg={self.phase:.2f}, "
            f"east_km={self.east_km:.2f}, north_km={self.north_km:.2f})"
        )

    def __str__(self) -> str:
        return f"{self.magnitude:.2f} km @ {self.phase:.2f}°"

    def __format__(self, format_spec: str) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Geodetic geometry — GeoPoint
# ---------------------------------------------------------------------------


class GeoPoint(tuple):
    """A point on the Earth's surface, stored as (latitude, longitude) in decimal degrees.

    Arithmetic follows the same algebraic structure as :class:`Position`:

    * ``GeoPoint + GeoVector  → GeoPoint``   (Vincenty direct formula)
    * ``GeoPoint − GeoPoint   → GeoVector``  (Vincenty inverse formula)
    """

    def __new__(cls, lat: float, long: float) -> GeoPoint:
        return tuple.__new__(cls, (lat, long))

    @property
    def latitude(self) -> float:
        return self[0]

    @property
    def longitude(self) -> float:
        return self[1]

    def __add__(self, vector: GeoVector) -> GeoPoint:
        """Displace this point along *vector* using the Vincenty direct formula."""
        distance, bearing_radians = cmath.polar(vector)
        lat, long, _ = _vincdir(
            self.latitude, self.longitude, degrees(bearing_radians), distance
        )
        # Wrap longitude to (-180, 180]
        long = (long + 180) % 360 - 180
        if long == -180:
            long = 180.0
        return GeoPoint(lat, long)

    def __sub__(self, other: GeoPoint) -> GeoVector:
        """Return the geodetic displacement from *other* to *self* (Vincenty inverse)."""
        dist, _fwd_az, rev_az = _vincinv(
            self.latitude,
            self.longitude,
            other.latitude,
            other.longitude,
        )
        return GeoVector(dist / 1000, rev_az)  # dist → km

    def rotate_around_point(self, pivot: GeoPoint, angle_in_degree: float) -> GeoPoint:
        """Return this point rotated by *angle_in_degree* degrees around *pivot*."""
        vec = self - pivot
        return pivot + vec.turn(angle_in_degree)

    # -- point-in-region tests -----------------------------------------------

    def is_point_inside_triangle(self, A: GeoPoint, B: GeoPoint, C: GeoPoint) -> bool:
        """Return ``True`` if this point lies inside triangle ABC.

        Uses barycentric coordinates computed from Vincenty-derived vectors,
        so it remains accurate over medium distances on the Earth's surface.
        """
        v0: GeoVector = C - A
        v1: GeoVector = B - A
        v2: GeoVector = self - A

        dot00 = v0.real**2 + v0.imag**2  # |v0|²
        dot01 = v0.real * v1.real + v0.imag * v1.imag
        dot02 = v0.real * v2.real + v0.imag * v2.imag
        dot11 = v1.real**2 + v1.imag**2  # |v1|²
        dot12 = v1.real * v2.real + v1.imag * v2.imag

        denominator = dot00 * dot11 - dot01**2
        if denominator == 0:  # degenerate triangle
            return False

        inv_denom = 1.0 / denominator
        alpha = (dot11 * dot02 - dot01 * dot12) * inv_denom
        beta = (dot00 * dot12 - dot01 * dot02) * inv_denom
        gamma = 1.0 - alpha - beta

        return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

    def is_point_inside_polygon(self, polygon: list[GeoPoint]) -> bool:
        """Return ``True`` if this point lies inside *polygon* (ray-casting algorithm).

        The polygon vertices must be supplied in order (clockwise or
        counter-clockwise).  The test casts a ray eastward (increasing
        longitude) and counts edge crossings.
        """
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            lat_i, lon_i = polygon[i].latitude, polygon[i].longitude
            lat_j, lon_j = polygon[j].latitude, polygon[j].longitude

            crosses_latitude_band = (lon_i < self.longitude <= lon_j) or (
                lon_j < self.longitude <= lon_i
            )
            if crosses_latitude_band and lon_i != lon_j:
                intersect_lat = lat_i + (self.longitude - lon_i) / (lon_j - lon_i) * (
                    lat_j - lat_i
                )
                if self.latitude < intersect_lat:
                    inside = not inside
            j = i

        return inside

    # -- constructors --------------------------------------------------------

    @staticmethod
    def construct(human_latlong: str) -> GeoPoint:
        """Parse a ``'DDMMSSN DDDMMSSD'`` string and return the corresponding GeoPoint.

        Example::

            GeoPoint.construct('505405N 0042904E')   # Brussels airport
        """
        lat_str, long_str = human_latlong.split()
        return GeoPoint(dms2dec(lat_str), dms2dec(long_str, is_longitude=True))

    @staticmethod
    def construct_from_name(point_name: str) -> GeoPoint:
        """Look up an aerodrome by ICAO code and return its :class:`GeoPoint`.

        Requires ``my_aerodromes`` to be importable.
        """
        import my_aerodromes as airports

        coord = airports.coordinates[point_name][0]
        return GeoPoint.construct(coord)

    # -- display -------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GeoPoint(lat={self.latitude:.6f}, long={self.longitude:.6f})"

    def __str__(self) -> str:
        return (
            f'{dd2dms(self.latitude, "NS")} '
            f'{dd2dms(self.longitude, "EW")} '
            f'[{self.latitude:.2f}, {self.longitude:.2f}]'
        )

    def __format__(self, format_spec: str) -> str:
        match format_spec:
            case "h":  # human-readable DMS
                return f'{dd2dms(self.latitude, "NS")} {dd2dms(self.longitude, "EW")}'
            case "d":  # decimal degrees
                return f"[{self.latitude:.2f}, {self.longitude:.2f}]"
            case _:
                return str(self)


# ---------------------------------------------------------------------------
# Smoke tests / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import folium
    from my_aerodromes import coordinates  # noqa: F401  (needed by construct_from_name)

    # -- GeoPoint arithmetic demo --------------------------------------------
    p1 = GeoPoint.construct_from_name("EBBR")  # Brussels airport
    p2 = GeoPoint.construct_from_name("LFPG")  # Paris Charles-de-Gaulle

    print(f"EBBR: {p1}")
    print(f"LFPG: {p2}")

    vec = p2 - p1
    print(f"Displacement EBBR→LFPG: {vec}")

    middle = p1 + (p2 - p1) / 2
    p3 = p1 + GeoVector(distance_km=100, bearing_deg=-90)  # 100 km due west of EBBR

    point_within = middle + GeoVector(distance_km=25, bearing_deg=-90)
    point_outside = middle + GeoVector(distance_km=60, bearing_deg=-90)

    assert point_within.is_point_inside_triangle(p1, p2, p3), "Expected inside triangle"
    assert not point_outside.is_point_inside_triangle(
        p1, p2, p3
    ), "Expected outside triangle"
    assert point_within.is_point_inside_polygon([p1, p2, p3]), "Expected inside polygon"
    assert not point_outside.is_point_inside_polygon(
        [p1, p2, p3]
    ), "Expected outside polygon"
    print("All point-in-region assertions passed.")

    # -- Folium map ----------------------------------------------------------
    m = folium.Map(location=middle, zoom_start=6, control_scale=True)

    for location, label in [
        (p1, "EBBR (Brussels)"),
        (p2, "LFPG (Paris CDG)"),
        (p3, "100 km west of EBBR"),
        (middle, "Midpoint EBBR–LFPG"),
        (point_within, "25 km west of midpoint (inside)"),
        (point_outside, "60 km west of midpoint (outside)"),
    ]:
        folium.Marker(location=location, tooltip=label).add_to(m)

    folium.Polygon(locations=[p1, p2, p3]).add_to(m)
    m.fit_bounds([p1, p2])
    m.show_in_browser()

    # -- flat-plane Vector demo ----------------------------------------------
    v1 = Vector(1, 30) * 5
    v2 = Vector(2, 40) * 5
    print(f"\nv1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 rotated 20° = {v1.rotated_by(20)}")
