import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
from PIL import Image
import numpy as np

DATA_DIR = "/home/teddy/Desktop/Omniverse_test/replicator_data/run_20251118_222049"
HDF5_FILE = "robot_state_data.hdf5"

# 讀取資料
hdf5_path = os.path.join(DATA_DIR, HDF5_FILE)
if not os.path.exists(hdf5_path):
    print(f"錯誤: 找不到檔案 {hdf5_path}")
    exit()

f = h5py.File(hdf5_path, 'r')
# 取得所有 step 的 key 並排序 (確保順序正確: step_0000, step_0001...)
keys = sorted(list(f.keys()))
total_frames = len(keys)
print(f"成功載入，共 {total_frames} 幀資料。")

# 設定 Matplotlib 介面佈局
fig = plt.figure(figsize=(16, 8))
plt.subplots_adjust(bottom=0.25) # 預留底部空間給滑桿
# --- 建立子圖 (Subplots) ---
# 手腕相機
ax_wrist = fig.add_subplot(1, 3, 1)
ax_wrist.set_title("Wrist Camera")
ax_wrist.axis('off')
# 主相機
ax_main = fig.add_subplot(1, 3, 2)
ax_main.set_title("Main Camera (World)")
ax_main.axis('off')
# 文字資訊
ax_text = fig.add_subplot(1, 3, 3)
ax_text.set_title("Robot State")
ax_text.axis('off')
# 初始化顯示物件 (先顯示第一張圖，之後用 update 更新它)
# 讀取第一幀的路徑
first_step = f[keys[0]]
# 注意：從 HDF5 讀出的字串可能是 bytes，需要 decode
path_w = first_step['image_path_wrist'][()].decode('utf-8') if hasattr(first_step['image_path_wrist'][()], 'decode') else str(first_step['image_path_wrist'][()])
path_m = first_step['image_path_main'][()].decode('utf-8') if hasattr(first_step['image_path_main'][()], 'decode') else str(first_step['image_path_main'][()])
# 載入圖片
img_w = Image.open(os.path.join(DATA_DIR, path_w))
img_m = Image.open(os.path.join(DATA_DIR, path_m))
# 顯示圖片物件
im_display_w = ax_wrist.imshow(img_w)
im_display_m = ax_main.imshow(img_m)
# 顯示文字物件
text_display = ax_text.text(0.05, 0.95, "", transform=ax_text.transAxes, verticalalignment='top', fontfamily='monospace', fontsize=10)

# 定義更新函式 (核心邏輯)
def update(val):
    # 1. 取得滑桿當前的數值 (Frame Index)
    idx = int(slider_frame.val)
    step_key = keys[idx]
    
    # 2. 讀取該幀的資料
    group = f[step_key]
    
    # 處理路徑 (HDF5 字串解碼)
    p_wrist = group['image_path_wrist'][()]
    p_main = group['image_path_main'][()]
    # 確保轉成一般的 string
    if isinstance(p_wrist, bytes): p_wrist = p_wrist.decode('utf-8')
    if isinstance(p_main, bytes): p_main = p_main.decode('utf-8')
        
    full_p_wrist = os.path.join(DATA_DIR, p_wrist)
    full_p_main = os.path.join(DATA_DIR, p_main)
    
    # 讀取數值資料
    ts = group['timestamp'][()]
    joints = group['joint_pos'][:]
    ee_pos = group['ee_pose_pos'][:]
    ee_rot = group['ee_pose_rot_quat'][:]
    
    # 3. 更新圖片
    # 使用 set_data 比重新 imshow 快非常多
    if os.path.exists(full_p_wrist):
        im_display_w.set_data(Image.open(full_p_wrist))
    if os.path.exists(full_p_main):
        im_display_m.set_data(Image.open(full_p_main))
        
    # 4. 更新文字內容
    info_text = (
        f"Frame Index: {idx} / {total_frames-1}\n"
        f"Key Name: {step_key}\n"
        f"Timestamp: {ts:.4f} s\n\n"
        
        f"--- Joint Positions (Rad) ---\n"
        f"J1: {joints[0]:.4f}\n"
        f"J2: {joints[1]:.4f}\n"
        f"J3: {joints[2]:.4f}\n"
        f"J4: {joints[3]:.4f}\n"
        f"J5: {joints[4]:.4f}\n"
        f"J6: {joints[5]:.4f}\n"
        f"J7: {joints[6]:.4f}\n"
        f"Gripper L: {joints[7]:.4f}\n"
        f"Gripper R: {joints[8]:.4f}\n\n"
        
        f"--- EE Pose (World) ---\n"
        f"Pos X: {ee_pos[0]:.4f}\n"
        f"Pos Y: {ee_pos[1]:.4f}\n"
        f"Pos Z: {ee_pos[2]:.4f}\n\n"
        f"Rot (Quat): {np.round(ee_rot, 3)}"
    )
    text_display.set_text(info_text)
    
    # 5. 刷新畫布
    fig.canvas.draw_idle()

# 建立控制元件 (滑桿)
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_frame = Slider(
    ax=ax_slider,
    label='Frame Index',
    valmin=0,
    valmax=total_frames - 1,
    valinit=0,
    valstep=1
)

# 當滑桿移動時，呼叫 update 函式
slider_frame.on_changed(update)

# 第一次啟動時先執行一次 update 以顯示第一幀的文字
update(0)

print("視覺化視窗已啟動，請查看視窗。")
plt.show()

# 程式結束後關閉檔案
f.close()