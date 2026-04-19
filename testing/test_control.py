import mujoco 
import mujoco.viewer

import time

"""
CENTRES
U: YELLOW
D: WHITE
F: BLUE
B: GREEN
R: RED
L: ORANGE
"""

"""
CORNERS
PERM
0: (-1, -1, -1) LUF
1: (1, -1, -1) RUF
2 (-1, 1, -1) LDF
3: (1, 1, -1) RDF
4: (-1, -1, 1) LUB
5: (1, -1, 1) RUB
6: (-1, 1, 1) LDB
7: (1, 1, 1) RDB
ORIENT
0: NORMAL
1: CLOCKWISE
1: ANTICLOCKWISE
"""

"""
EDGES
0: (0, -1, -1) UF 
1: (-1, 0, -1) LF
2: (1, 0, -1) RF
3: (0, 1, -1) DF
4: (-1, -1, 0) UL
5: (1, -1, 0) UR
6: (-1, 1, 0) DL
7: (1, 1, 0) DR
8: (0, -1, 1) UB
9: (-1, 0, 1) LB
10: (1, 0, 1) RB
11: (0, 1, 1) DB
ORIENT
0: NORMAL
1: ABNORMAL
"""
    
def transform_x(pos, sign=True):
    x, y, z = pos
    if sign:
        return [x, -z, y]
    else:
        return [x, z, -y]

def transform_y(pos, sign=True):
    x, y, z = pos
    if sign:
        return [-z, y, x]
    else:
        return [z, y, -x]

def transform_z(pos, sign=True):
    x, y, z = pos
    if sign:
        return [y, -x, z]
    else:
        return [-y, x, z]

CORNERS = []
for z in [-1, 1]:
   for y in [-1, 1]:
      for x in [-1, 1]:
         CORNERS.append([x, y, z])

EDGES = []
for z in [-1, 0, 1]:
   for y in [-1, 0, 1]:
      for x in [-1, 0, 1]:
         if (x + y + z) % 2 == 0 and (x or y or z):
            EDGES.append([x, y, z])

class Cube:
    def __init__(self):
        self.CORNERS_PERM = CORNERS.copy()
        self.EDGES_PERM = EDGES.copy()

    def move(self, move_name):
        "redirect to proper channel"
        pass

    def U(self, inv):
        def transform(coords):
            if coords[1] == -1:
                coords = transform_y(coords, not inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]
    
    def D(self, inv: bool):
        def transform(coords):
            if coords[1] == 1:
                coords = transform_y(coords, inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]

    def R(self, inv: bool):
        def transform(coords):
            if coords[0] == 1:
                coords = transform_x(coords, not inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]
    
    def L(self, inv: bool):
        def transform(coords):
            if coords[0] == -1:
                coords = transform_x(coords, inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]

    def F(self, inv: bool):
        def transform(coords):
            if coords[2] == -1:
                coords = transform_z(coords, not inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]
    
    def B(self, inv: bool):
        def transform(coords):
            if coords[2] == 1:
                coords = transform_z(coords, inv)
            return coords
        
        self.CORNERS_PERM = [transform(corner) for corner in self.CORNERS_PERM]
        self.EDGES_PERM = [transform(edge) for edge in self.EDGES_PERM]


rub = Cube()
print(rub.CORNERS_PERM)
rub.U(False)
print(rub.CORNERS_PERM)

# m = mujoco.MjModel.from_xml_path('Cube.xml')
# d = mujoco.MjData(m)

# with mujoco.viewer.launch_passive(m, d) as viewer:
#   # Close the viewer automatically after 3-1 wall-seconds.
#   start = time.time()
#   while viewer.is_running() and time.time() - start < 3000:
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(m, d)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = m.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > -1:
#       time.sleep(time_until_next_step)