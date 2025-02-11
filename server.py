import gradio as gr
from typing import Optional, List
import os

from intelivid.keyframe_extraction import extract_keyframes
from intelivid.multimodal_understanding import main as understand_images
from intelivid.video_summarization import main as summarize_video
from intelivid.video_qa import main as video_qa
from intelivid.video_classification import main as classify_video


def process_video(video_path: str, mode: str, analysis_level: str, 
                 task: str, question: Optional[str] = None) -> str:
    # Step 1: Extract keyframes
    extract_keyframes(video_path, mode=mode)
    
    # Step 2: Analyze keyframes
    understand_images(analysis_level)
    
    # Step 3: Process based on selected task
    if task == "summary":
        return summarize_video()
    elif task == "qa" and question:
        return video_qa(question)
    elif task == "classification":
        return classify_video()
    elif task == "semantic_search":
        try:
            from video_semantic_search import main as semantic_search
            if not os.path.exists("output/captions/basic"):
                return "请先进行视频分析以生成所需目录"
            results = semantic_search(question)
            return format_search_results(results)
        except ImportError:
            return "语义搜索模块加载失败"
        except Exception as e:
            return f"语义搜索出错: {str(e)}"
    else:
        return "请选择有效的任务类型"

def format_search_results(docs: List) -> List[tuple]:
    """格式化搜索结果并返回(图片路径, 字幕内容)元组列表"""
    results = []
    for doc in docs:
        caption_path = doc.metadata['source']
        # 将字幕路径转换为对应的关键帧图片路径
        keyframe_path = caption_path.replace('captions\\basic\\', 'keyframes\\').replace('.txt', '.png')
        if os.path.exists(keyframe_path):
            # 读取字幕内容
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_text = f.read()
            results.append((keyframe_path, caption_text))
    return results


def chatbot_interface(video_path, question, chat_history):
    response = process_video(video_path, "scenedetect", "semantic", "qa", question)
    chat_history.append((question, response))
    return "", chat_history


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# InteliVid")
        
        with gr.Tabs():
            with gr.TabItem("视频摘要"):
                with gr.Row():
                    video_input = gr.Video(label="上传视频")
                    mode_select = gr.Dropdown(
                        choices=["ffmpeg", "scenedetect"],
                        value="scenedetect",
                        label="关键帧提取模式"
                    )
                    analysis_select = gr.Dropdown(
                        choices=["basic", "semantic", "emotional"],
                        value="basic",
                        label="图像理解层次"
                    )
                output_text = gr.Textbox(label="视频总结", lines=10)
                submit_btn = gr.Button("开始分析")
                submit_btn.click(
                    fn=process_video,
                    inputs=[video_input, mode_select, analysis_select, gr.State("summary")],
                    outputs=output_text
                )
            
            with gr.TabItem("视频问答"):
                with gr.Row():
                    video_input_qa = gr.Video(label="上传视频", sources=["upload"])
                chatbot = gr.Chatbot(label="对话历史")
                question_input = gr.Textbox(label="输入问题")
                submit_btn = gr.Button("提交")
                
                submit_btn.click(
                    fn=chatbot_interface,
                    inputs=[video_input_qa, question_input, chatbot],
                    outputs=[question_input, chatbot]
                )
                
                gr.Examples(
                    examples=["视频的主要内容是什么?", "视频中有哪些关键场景?"],
                    inputs=question_input
                )
            
            with gr.TabItem("视频分类"):
                with gr.Row():
                    video_input_cls = gr.Video(label="上传视频")
                output_cls = gr.Textbox(label="分类结果", lines=10)
                cls_btn = gr.Button("开始分类")
                cls_btn.click(
                    fn=process_video,
                    inputs=[video_input_cls, gr.State("scenedetect"), gr.State("semantic"), gr.State("classification")],
                    outputs=output_cls
                )
            
            with gr.TabItem("语义搜索"):
                with gr.Row():
                    video_input_search = gr.Video(label="上传视频")
                    search_query = gr.Textbox(label="搜索内容")
                output_search = gr.Gallery(label="搜索结果", columns=3, height="auto")
                search_btn = gr.Button("开始搜索")
                search_btn.click(
                    fn=process_video,
                    inputs=[video_input_search, gr.State("scenedetect"), gr.State("semantic"), gr.State("semantic_search"), search_query],
                    outputs=output_search
                )
        
    demo.launch()

if __name__ == "__main__":
    main()
