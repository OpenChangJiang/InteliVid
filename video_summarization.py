import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import intelivid.i18n.zh.prompts as prompts


def read_captions(directory):
    """Read all caption files from a directory and return as a list"""
    captions = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                captions.append(f.read().strip())
    return captions

def generate_summary_prompt(basic_captions):
    """Generate prompt for summarizing video characteristics"""
    prompt = prompts.VIDEO_SUMMARY
    return prompt.format(
        basic='\n'.join(f"第{i+1}帧: {caption}" for i, caption in enumerate(basic_captions))
    )

def main():
    # Read basic captions
    base_dir = './output/captions'
    basic_captions = read_captions(os.path.join(base_dir, 'basic'))

    # Generate summary prompt
    prompt = generate_summary_prompt(basic_captions)

    # Initialize API client
    client = OpenAI(
        api_key=os.getenv("BAIDUBCE_API_KEY"),
        base_url="https://qianfan.baidubce.com/v2",
    )

    # Get summary from deepseek-v3
    completion = client.chat.completions.create(
        model="deepseek-v3",
        messages=[
            {'role': 'system', 'content': 'You are a professional video editor.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    # Return the summary
    return completion.choices[0].message.content


if __name__ == "__main__":
    main()
