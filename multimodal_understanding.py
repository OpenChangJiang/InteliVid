import base64
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

import intelivid.i18n.zh.prompts as zh_prompts

# Load model once at module level
model_path = "./models/Janus-Pro-7B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    model_path, 
    use_fast=True
)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)

def run_inference(image_path, level, output_dir, prompt):
    # 将图片转换为base64字符串
    buffered = BytesIO()
    Image.open(image_path).save(buffered, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    
    # 如果描述文件已存在则跳过
    caption_file = output_dir / f"{image_path.stem}.png"
    if caption_file.exists():
        return
        
    # 构建对话内容
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 加载图片并准备输入
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # 运行图像编码器以获取图像嵌入
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # 运行模型以获取响应
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    # 解码并输出回答
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
    # 保存描述到txt文件
    with open(caption_file.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(answer)
            
    print(f"Processed {image_path.name} -> {caption_file.stem}.txt")

def main(analysis_level):

    # 创建不同层次的输出目录
    output_dirs = {
        "basic": Path("./output/captions/basic"),
        "semantic": Path("./output/captions/semantic"), 
        "emotional": Path("./output/captions/emotional")
    }
    for dir in output_dirs.values():
        dir.mkdir(parents=True, exist_ok=True)

    # 定义不同层次的Prompt
    prompts = {
        "basic": zh_prompts.IMAGE_ANALYSIS_BASIC,
    }

    # 遍历keyframes目录下的所有图片
    from tqdm import tqdm
    keyframes_dir = Path("./output/keyframes")
    for image_path in tqdm(list(keyframes_dir.glob("*.png")), desc="Processing images"):
        # 对每个层次进行处理
        for level, prompt in prompts.items():
            run_inference(image_path, level, output_dirs[level], prompt)


if __name__ == "__main__":
    main('basic')
