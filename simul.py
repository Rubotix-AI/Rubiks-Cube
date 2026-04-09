from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.arenas import TableArena

class Cube(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("Cube.xml", name=name, joints="default", obj_type="all", duplicate_collision_geoms=True)

world = MujocoWorldBase()
muj_bot = Panda()


muj_bot.set_base_xpos([0, 0, 0])
world.merge(muj_bot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

cube_mod = Cube(
    name="cube"
).get_obj()

cube_mod.set('pos', '1.0 0 1.0')
world.worldbody.append(cube_mod)

model = world.get_model(mode="mujoco")

import mujoco

data = mujoco.MjData(model)
while data.time < 1:
    mujoco.mj_step(model, data)