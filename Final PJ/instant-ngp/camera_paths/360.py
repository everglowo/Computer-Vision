import json
import re
from pathlib import Path

def to_homogeneous(c2w):
    """Ensure camera_to_world is a 4x4 homogeneous matrix."""
    if len(c2w) == 3:
        c2w.append([0, 0, 0, 1])
    return c2w

# 修复点：原始 JSON 是 list，不含 "frames" 键
with open('transforms_train.json') as f:
    train_transforms = json.load(f)

with open('transforms_eval.json') as f:
    eval_transforms = json.load(f)

# 提取帧编号用于排序
def extract_frame_number(filepath):
    stem = Path(filepath).stem
    match = re.search(r'\d+', stem)
    return int(match.group()) if match else 0

all_frames = train_transforms + eval_transforms
all_frames = sorted(all_frames, key=lambda x: extract_frame_number(x["file_path"]))

# 构造 camera_path.json
output = {
    'camera_type': 'perspective',
    'render_height': 1080,
    'render_width': 1920,
    'seconds': len(all_frames) / 24,  # 假设 24 fps
    'camera_path': [
        {
            'camera_to_world': to_homogeneous(pose['transform']),
            'fov': 50,
            'aspect': 1.0,
            'file_path': pose['file_path']
        }
        for pose in all_frames
    ]
}

# 写入 JSON 文件
with open('camera_path.json', 'w') as f:
    json.dump(output, f, indent=4)

print("✅ camera_path.json 已成功生成！")
