from shapely import Polygon, Point
from corridor_opt.geometry import GeometryWrapper
from typing import Tuple
from shapely import affinity
import numpy as np

class Obstacle(GeometryWrapper):
    def __init__(self, xy: list | None = None, polygon: Polygon | None = None, geometry_type: type=Polygon, img: str | None = None, id: int | None = None, depth: float | None = None, color: str = 'black'):
        self._id = id
        self._depth = depth
        self.color = color
        super().__init__(xy=xy, polygon=polygon, geometry_type=geometry_type)

    def distance_to_obstacle(self, x:float, y:float) -> float:
        """
        Get the distance to the obstacle at a given position.
        """
        assert isinstance(self._geometry, Polygon), f"Obstacle must be a polygon not a {type(self._geometry)}"
        p = Point([x, y])
        if p.within(self._geometry):
            return -p.distance(self._geometry.exterior)
        return p.distance(self._geometry.exterior)
    
    def __repr__(self):
        return f"Obstacle({self.centroid[0]:.2f}, {self.centroid[1]:.2f})"
    
    def plot(self, *args, ax=None, c=None, offset: Tuple | None = None, **kwargs):
        """
        Plot the obstacle.
        """
        return super().plot(*args, ax=ax, c=c or self.color, offset=offset, **kwargs)
    
    def fill(self, *args, ax=None, c=None, **kwargs):
        """
        Fill the obstacle.
        """
        return super().fill(*args, ax=ax, c=c or self.color, **kwargs)
    
    @property
    def id(self) -> int | None:
        return self._id
    
    @property
    def depth(self) -> float | None:
        return self._depth
    
class Ellipse(Obstacle):
    def __init__(self,
                x: float,
                y: float,
                a: float,
                b: float,
                id: int | None = None
                ):
        super().__init__(polygon=affinity.translate(affinity.scale(Point([0, 0]).buffer(1), b, a), x, y), id=id)
        self._a = a
        self._b = b
        self._da = x
        self._db = y

    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b
    
    @property
    def center(self):
        return self.centroid
    
    @center.setter
    def center(self, value:Tuple):
        self.centroid = value

    @property
    def da(self) -> float:
        return self._da
    
    @property
    def db(self) -> float:
        return self._db
    
    def __repr__(self):
        return f"Ellipse({self.a:.2f}, {self.b:.2f} at {self.center[0]:.2f}, {self.center[1]:.2f})"
    
class Circle(Ellipse):
    def __init__(self, x, y, radius, id: int | None =None):
        super().__init__(x, y, radius, radius, id=id)
        self._radius = radius

    def __repr__(self):
        return f"Circle(at {self.center[0]:.2f}, {self.center[1]:.2f} with radius {self.radius:.2f})"

    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        self._radius = value
        self._geometry = affinity.scale(Point(self.center).buffer(1), value, value)

class Rectangle(Obstacle):
    def __init__(self, x, y, height, width, id:int | None = None, *args, **kwargs): # 3 -5 2 4
        assert height > 0, f"height of {type(self).__name__} object must be > 0. Got {height}"
        assert width > 0, f"width of {type(self).__name__} object must be > 0. Got {width}"
        self._center = (x, y)
        self._dim = (width, height)
        super().__init__(xy=self.get_envelope_coordinates(), *args, **kwargs)
        
    def get_envelope_coordinates(self) -> list[Tuple]:
        # Compute vertices coordinates
        v1 = (self._center[0] + self._dim[0]/2 , self._center[1] + self._dim[1] / 2)
        v2 = (self._center[0] + self._dim[0]/2 , self._center[1] - self._dim[1] / 2)
        v3 = (self._center[0] - self._dim[0]/2 , self._center[1] - self._dim[1] / 2)
        v4 = (self._center[0] - self._dim[0]/2 , self._center[1] + self._dim[1] / 2)
        return [v1, v2, v3, v4]
    
    @property
    def lower_right_corner(self) -> np.ndarray:
        return np.array(self._geometry.exterior.coords[1])
    
    @property
    def lower_left_corner(self) -> np.ndarray:
        return np.array(self._geometry.exterior.coords[2])
    
    @property
    def upper_right_corner(self) -> np.ndarray:
        return np.array(self._geometry.exterior.coords[0])
    
    @property
    def upper_left_corner(self) -> np.ndarray:
        return np.array(self._geometry.exterior.coords[3])
    
    @property
    def height(self) -> float:
        return self._dim[1]
    
    @property
    def width(self) -> float:
        return self._dim[0]
