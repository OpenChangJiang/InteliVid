from scenedetect import detect, AdaptiveDetector
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from pathlib import Path
import csv


def extract_keyframes(video_path, output_dir, metadata_file):
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 使用 PySceneDetect 检测场景变化
    scene_list = detect(video_path, AdaptiveDetector())
    
    # 使用 MoviePy 加载视频
    clip = VideoFileClip(video_path)
    
    keyframe_files = []
    # 提取每个场景的关键帧
    for i, scene in enumerate(scene_list):
        start_time, end_time = scene
        key_frame = clip.get_frame(start_time.get_seconds())  # 提取场景开始时的帧
        frame_path = output_path / f"scene_{i:04d}_keyframe.png"
        Image.fromarray(key_frame).save(frame_path)
        keyframe_files.append(frame_path)
    
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
