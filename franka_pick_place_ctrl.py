from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core.utils.stage import open_stage
import omni.usd
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction



USD_PATH = "/home/teddy/Desktop/Omniverse_test/franka_pick_test.usd"
# omni.usd.get_context().open_stage(USD_PATH)
open_stage(USD_PATH)

world = World(stage_units_in_meters=1.0)

FRANKA_PATH = "/Root/franka"                  # 場景裡的路徑
CUBE_PATH   = "/Root/Cube"
LONG_SCISSOR_PATH = "/Root/long_scissor"
SCISSORS_PATH = "/Root/scissors"
BANANA_PATH = "/Root/banana"
GRIP_PATH = "/Root/grip"

ee_path = "/Root/franka/panda_hand"         # 末端 prim

from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from omni.isaac.core.prims import XFormPrim 

gripper = ParallelGripper(
    end_effector_prim_path=f"{FRANKA_PATH}/panda_hand",  # 末端
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04]),
    joint_closed_positions=np.array([0.00]),
    action_deltas=np.array([0.017]),
    use_mimic_joints=True
)

franka = world.scene.add(
    SingleManipulator(
        prim_path=FRANKA_PATH,
        name="franka",
        end_effector_prim_path=f"{FRANKA_PATH}/panda_hand",
        gripper=gripper,
    )
)

long_scissor = world.scene.add(XFormPrim(prim_path=LONG_SCISSOR_PATH, name="long_scissor"))
scissors = world.scene.add(XFormPrim(prim_path=SCISSORS_PATH, name="scissors"))
banana = world.scene.add(XFormPrim(prim_path=BANANA_PATH, name="banana"))
grip = world.scene.add(XFormPrim(prim_path=GRIP_PATH, name="grip"))

# target = banana
# position, orientation = target.get_world_pose()
# print(position)
# pick_pose = position.copy()
# PLACE_Z = position[2]
# place_pos = np.array([0.30, -0.30, PLACE_Z])
# ee_offset = np.array([0.0, 0.0, -0.01])

# scissor: action_deltas=np.array([0.017])
target = scissors
position, orientation = target.get_world_pose()
print(position)
pick_pose = position.copy()
PLACE_Z = position[2] + 0.01
place_pos = np.array([0.30, -0.30, PLACE_Z])
ee_offset = np.array([0.0, 0.0, 0.008])

# long_scissor: action_deltas=np.array([0.019])
# target = long_scissor
# position, orientation = target.get_world_pose()
# print(position)
# pick_pose = position.copy()
# PLACE_Z = position[2] + 0.01
# place_pos = np.array([0.30, -0.30, PLACE_Z])
# ee_offset = np.array([0.0, -0.002, 0.005])

# grip: action_deltas=np.array([0.0177])
# target = grip
# position, orientation = target.get_world_pose()
# print(position)
# pick_pose = position.copy()
# PLACE_Z = position[2] + 0.01
# place_pos = np.array([0.30, -0.30, PLACE_Z])
# ee_offset = np.array([0.0, 0.0, 0.005])

franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

""""
    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.
"""
controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    events_dt=[0.008, 0.005, 1, 0.1, 0.005, 0.01, 0.005, 1, 0.08, 0.08],
)
articulation_controller = franka.get_articulation_controller()

reset_needed = False
task_done = False

while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True
        task_done = False
    
    if world.is_playing():
        if reset_needed:
            world.reset()
            controller.reset()
            reset_needed = False
            task_done = False

        obs = world.get_observations()

        actions = controller.forward(
            picking_position=pick_pose,
            placing_position=place_pos,
            current_joint_positions=franka.get_joint_positions(),
            end_effector_offset=ee_offset,
        )

        if controller.is_done() and not task_done:
            print("[franka_pick_up] done picking and placing")
            task_done = True
        
        articulation_controller.apply_action(actions)
    
simulation_app.close()