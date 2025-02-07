import subprocess
from pathlib import Path
import csv


def extract_keyframes(video_path, output_dir, metadata_file):
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用ffmpeg直接提取所有关键帧
    ffmpeg_cmd = f'ffmpeg -i "{video_path}" -vf "select=eq(pict_type\\,I)" -vsync vfr "{output_path}/keyframe_%04d.png"'
    
    print("执行FFmpeg命令:", ffmpeg_cmd)
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg错误: {e.stderr}")
        return

    # 获取提取的关键帧文件
    keyframe_files = sorted(output_path.glob("keyframe_*.png"))
    
    if not keyframe_files:
        print("未检测到关键帧")
        return

    # 记录元数据
    _write_metadata(output_path, metadata_file, keyframe_files)

def _write_metadata(output_path, metadata_file, keyframe_files):
    metadata_path = output_path / metadata_file
    with open(metadata_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # 写入表头
            writer.writerow(["Filename", "FrameNumber"])
            
        # 记录元数据
        for idx, frame_file in enumerate(keyframe_files, start=1):
            writer.writerow([
                frame_file.name,
                idx
            ])
        print(f"已提取{len(keyframe_files)}个关键帧")
