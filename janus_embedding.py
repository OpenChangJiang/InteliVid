import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
from io import BytesIO
import base64

class JanusEmbedder:
    def __init__(self, model_path):
        # Initialize model and processor
        self.vl_chat_processor = VLChatProcessor.from_pretrained(
            model_path,
            use_fast=True
        )
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load model
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
    
    def encode_text(self, text):
        """Encode text into embedding vector"""
        with torch.no_grad():
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Move inputs to GPU
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get embeddings from the model's language model
            return self.vl_gpt.language_model.get_input_embeddings()(inputs["input_ids"])
    
    def encode_image(self, image_path):
        """Encode image into embedding vector"""
        # Convert image to base64
        buffered = BytesIO()
        Image.open(image_path).save(buffered, format="PNG")
        image = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
        
        # Create conversation with image
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load image and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(self.vl_gpt.device)
        
        # Get image embeddings
        with torch.no_grad():
            return self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    def calculate_similarity(self, text_emb, image_emb):
        """Calculate cosine similarity between text and image embeddings"""
        # Average embeddings across sequence length
        text_emb = text_emb.mean(dim=1)
        image_emb = image_emb.mean(dim=1)
        
        # Calculate cosine similarity
        return F.cosine_similarity(text_emb, image_emb).item()


def main():
    model_path = "./models/Janus-Pro-1B"
    embedder = JanusEmbedder(model_path=model_path)
    
    # Example text and image
    text = "A cat"
    text = "橘猫，在木地板上"
    text = "An orange cat"
    text = "An orange cat is on the wooden floor."
    # image_path = "./output/keyframes/scene_0000_keyframe.png"
    image_path = "./cat.png"
    
    # Encode both text and image
    text_emb = embedder.encode_text(text)
    image_emb = embedder.encode_image(image_path)
    
    # Calculate and print similarity
    similarity = embedder.calculate_similarity(text_emb, image_emb)
    print(f"Text-Image Similarity: {similarity:.4f}")

if __name__ == "__main__":
    main()
