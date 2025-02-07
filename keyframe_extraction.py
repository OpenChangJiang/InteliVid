import argparse

from intelivid.extractors import scenedetect, ffmpeg


def extract_keyframes(video_path, mode='scenedetect', output="output/keyframes", metadata="metadata.csv"):
    if mode == 'ffmpeg':
        ffmpeg.extract_keyframes(
            video_path=video_path,
            output_dir=output,
            metadata_file=metadata
        )
    else:
        scenedetect.extract_keyframes(
            video_path=video_path,
            output_dir=output,
            metadata_file=metadata
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='视频关键帧提取工具')
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('-m', '--mode', choices=['ffmpeg', 'scenedetect'], 
                       default='scenedetect', help='提取模式:ffmpeg或scenedetect')
    parser.add_argument('-o', '--output', help='输出目录', default="output/keyframes")
    parser.add_argument('--metadata', help='元数据文件名', default="metadata.csv")
    
    args = parser.parse_args()
    extract_keyframes(args.video_path, args.mode, args.output, args.metadata)
