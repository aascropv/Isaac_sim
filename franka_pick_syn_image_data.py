from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core.utils.stage import open_stage
import omni.usd
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
import omni.replicator.core as rep
import os
import datetime

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

wrist_cam_path = f"{ee_path}/franka_camera" # 相機 prim
main_cam_path = "/Root/Camera"

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

# 1. 為相機 "路徑" 建立 "Render Product" (渲染產品)
wrist_rp = rep.create.render_product(wrist_cam_path, (512, 512), name="wrist_view")
main_rp = rep.create.render_product(main_cam_path, (512, 512), name="main_view")
all_rps = [wrist_rp, main_rp]

# 2. 設定 output_dir
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"run_{run_timestamp}"
base_output_path = "/home/teddy/Desktop/Omniverse_test/replicator_data"
output_dir = os.path.abspath(os.path.join(base_output_path, run_folder_name))
print(f"將資料儲存到: {output_dir}")

# 3. 從註冊表 (Registry) 獲取 "Writer" "實例"
writer_type = "BasicWriter"
writer = rep.WriterRegistry.get(writer_type)

# 4. "初始化" Writer, 告訴它我們想要什麼資料 (rgb=True)
writer_kwargs = {
    "output_dir": output_dir,
    "rgb": True
}
writer.initialize(**writer_kwargs)

# 5. 將 Writer "附加" 到 "Render Products"
writer.attach(all_rps)

# 告訴 Replicator 自動將 "writer" 掛鉤到 "world.step()"
# 當您按下 "Play" (即 world.step() 開始運行), 
# Replicator 將 "自動" 在每一幀儲存影像。
rep.orchestrator.set_capture_on_play(True)

# --- 預設 "禁用" 所有 Render Product ---
# 這樣 Replicator 就不會在每一幀都擷取
print("預設禁用 Render Products...")
for rp in all_rps:
    rp.hydra_texture.set_updates_enabled(False)

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

# --- 影像擷取計數器 ---
step_counter = 0
# 每 10 幀物理步驟, 儲存一次影像
CAPTURE_INTERVAL = 10

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

            # --- 重置計數器 ---
            step_counter = 0
            # 確保重置時 RP 是禁用的
            for rp in all_rps:
                rp.hydra_texture.set_updates_enabled(False)

        obs = world.get_observations()

        actions = controller.forward(
            picking_position=pick_pose,
            placing_position=place_pos,
            current_joint_positions=franka.get_joint_positions(),
            end_effector_offset=ee_offset,
        )

        # 1. 檢查是否是我們 "想要" 擷取的那一幀
        if step_counter % CAPTURE_INTERVAL == 0:
            # 啟用 Render Products
            # set_capture_on_play(True) 會自動偵測到它們
            # 並在 "這一幀" 進行擷取
            print(f"Frame {step_counter}: 啟用擷取")
            for rp in all_rps:
                rp.hydra_texture.set_updates_enabled(True)
        # 2. 檢查是否是擷取後的 "下一幀"
        elif step_counter % CAPTURE_INTERVAL == 3:
            # 立刻將其禁用, 這樣在接下來的幀
            # Replicator 都會跳過它們, 模擬器將全速運行
            print(f"Frame {step_counter}: 禁用擷取")
            for rp in all_rps:
                rp.hydra_texture.set_updates_enabled(False)
        
        step_counter += 1

        if controller.is_done() and not task_done:
            print("[franka_pick_up] done picking and placing")
            task_done = True
        
        articulation_controller.apply_action(actions)
    
# --- 任務結束時, 清理 Replicator 元件 ---
writer.detach()
wrist_rp.destroy()
main_rp.destroy()

simulation_app.close()