import mujoco
import mujoco.viewer
import time
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.arenas import TableArena

world = MujocoWorldBase()
muj_bot = Panda()


muj_bot.set_base_xpos([0, 0, 0])
world.merge(muj_bot)

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])
world.merge(mujoco_arena)

cube = mujoco.MjModel.from_xml_path('Cube.xml')
world.worldbody.append(cube)

model = world.get_model(mode="mujoco")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 300:
        step_start = time.time()

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)