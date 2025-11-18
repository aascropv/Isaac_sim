from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core.utils.stage import open_stage
import omni.usd
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
import os
import datetime
import h5py
from PIL import Image

from omni.isaac.sensor import Camera
from omni.isaac.core.simulation_context import SimulationContext

USD_PATH = "/home/teddy/Desktop/Omniverse_test/franka_pick_test.usd"
# omni.usd.get_context().open_stage(USD_PATH)
open_stage(USD_PATH)

world = World(stage_units_in_meters=1.0)
simulation_context = SimulationContext()

FRANKA_PATH = "/Root/franka"                  # 場景裡的路徑
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
pick_pos = position.copy()
PLACE_Z = position[2] + 0.01
place_pos = np.array([0.30, -0.30, PLACE_Z])
ee_offset = np.array([0.0, 0.0, 0.008])

# long_scissor: action_deltas=np.array([0.019])
# target = long_scissor
# position, orientation = target.get_world_pose()
# print(position)
# pick_pos = position.copy()
# PLACE_Z = position[2] + 0.01
# place_pos = np.array([0.30, -0.30, PLACE_Z])
# ee_offset = np.array([0.0, -0.002, 0.005])

# grip: action_deltas=np.array([0.0177])
# target = grip
# position, orientation = target.get_world_pose()
# print(position)
# pick_pos = position.copy()
# PLACE_Z = position[2] + 0.01
# place_po = np.array([0.30, -0.30, PLACE_Z])
# ee_offset = np.array([0.0, 0.0, 0.005])

# 初始化 Camera 物件
print(f"正在初始化攝影機: {wrist_cam_path}")
wrist_camera = Camera(
    prim_path=wrist_cam_path,
    resolution=(640, 480)
)
print(f"正在初始化攝影機: {main_cam_path}")
main_camera = Camera(
    prim_path=main_cam_path,
    resolution=(640, 480)
)

# 設定 output_dir
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"run_{run_timestamp}"
base_output_path = "/home/teddy/Desktop/Omniverse_test/replicator_data"
output_dir = os.path.abspath(os.path.join(base_output_path, run_folder_name))
os.makedirs(output_dir, exist_ok=True)
print(f"將資料儲存到: {output_dir}")
# 定義並建立影像儲存的子資料夾
wrist_image_dir = os.path.join(output_dir, "images_wrist")
main_image_dir = os.path.join(output_dir, "images_main")
os.makedirs(wrist_image_dir, exist_ok=True)
os.makedirs(main_image_dir, exist_ok=True)

franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

# 在 world.reset() 之後, 呼叫 initialize()
# 這能確保攝影機在模擬開始前準備就緒
wrist_camera.initialize()
main_camera.initialize()

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
    # events_dt=[0.008, 0.005, 1, 0.1, 0.05, 0.01, 0.005, 1, 0.08, 0.08],
)
articulation_controller = franka.get_articulation_controller()

# 用來儲存所有 "幀" 的資料
data_to_record = []
frame_index = 0
step_counter = 0
SAVE_INTERVAL = 20

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
            data_to_record = []
            frame_index = 0
            step_counter = 0
            
        obs = world.get_observations()

        actions = controller.forward(
            picking_position=pick_pos,
            placing_position=place_pos,
            current_joint_positions=franka.get_joint_positions(),
            end_effector_offset=ee_offset,
        )

        if step_counter % SAVE_INTERVAL == 0:
            # 1. 直接從攝影機獲取影像資料
            rgba_wrist = wrist_camera.get_rgba()
            rgba_main = main_camera.get_rgba()

            # 2. 檢查影像是否有效 (剛啟動時可能為 None 或 shape (0,))
            if rgba_wrist is not None and rgba_main is not None and rgba_wrist.shape[0] > 0 and rgba_main.shape[0] > 0:
                # 3. 只有在影像有效時, 才蒐集 *所有* 資料 (確保同步)
                # 獲取 "機器人狀態" (Robot State)
                current_joint_pos = franka.get_joint_positions()
                ee_pos, ee_rot = franka.end_effector.get_world_pose()
                current_time = simulation_context.current_time

                # --- 儲存影像到硬碟 ---
                # 定義檔名 (例如 0001, 0002)
                filename_wrist = f"wrist_{frame_index:04d}.png"
                filename_main = f"main_{frame_index:04d}.png"

                # 定義完整的儲存路徑
                save_path_wrist = os.path.join(wrist_image_dir, filename_wrist)
                save_path_main = os.path.join(main_image_dir, filename_main)

                # 定義要存入 HDF5 的"相對路徑" (方便未來讀取)
                relative_path_wrist = os.path.join("images_wrist", filename_wrist)
                relative_path_main = os.path.join("images_main", filename_main)

                # 使用 PIL 將 NumPy array 存成 PNG
                # (Isaac Sim 輸出是 RGBA, PNG 支援 RGBA)
                img_wrist_pil = Image.fromarray(rgba_wrist, 'RGBA')
                img_main_pil = Image.fromarray(rgba_main, 'RGBA')

                img_wrist_pil.save(save_path_wrist)
                img_main_pil.save(save_path_main)

                # 4. 將 *實際的影像陣列* 存入字典
                frame_data = {
                    "timestamp": current_time,
                    "image_path_wrist": relative_path_wrist,
                    "image_path_main": relative_path_main,
                    "joint_pos": current_joint_pos.flatten().copy(),
                    "ee_pose_pos": ee_pos.flatten().copy(),        
                    "ee_pose_rot_quat": ee_rot.flatten().copy()
                }

                # 將這幀的資料存入我們的 "暫存列表"
                data_to_record.append(frame_data)
                print(f"成功擷取同步資料, 第 {len(data_to_record)} 筆")

                frame_index += 1
        
        step_counter += 1

        if controller.is_done() and not task_done:
            print("[franka_pick_up] done picking and placing")
            task_done = True
            break

        articulation_controller.apply_action(actions)

# --- 儲存 HDF5 狀態資料 ---
# 必須在關閉 App 之前執行此操作
hdf5_path = os.path.join(output_dir, "robot_state_data.hdf5")
print(f"正在將 {len(data_to_record)} 幀的狀態資料儲存到: {hdf5_path}")

try:
    with h5py.File(hdf5_path, 'w') as f:
        for i, frame in enumerate(data_to_record):
            # 'i' (0, 1, 2...) 會完美匹配 'capture_frame_index'
            group = f.create_group(f"step_{i:04d}")
            group.create_dataset("timestamp", data=frame["timestamp"])
            group.create_dataset("image_path_wrist", data=str(frame["image_path_wrist"]))
            group.create_dataset("image_path_main", data=str(frame["image_path_main"]))
            group.create_dataset("joint_pos", data=frame["joint_pos"])
            group.create_dataset("ee_pose_pos", data=frame["ee_pose_pos"])
            group.create_dataset("ee_pose_rot_quat", data=frame["ee_pose_rot_quat"])
    print("HDF5 狀態資料儲存完畢")

except Exception as e:
    print(f"儲存 HDF5 失敗: {e}")

simulation_app.close()