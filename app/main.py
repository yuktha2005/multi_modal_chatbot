import gradio as gr
from model_utils import chat_with_model

def interface(user_text, image):
    img_path = image.name if image else None
    return chat_with_model(user_text, img_path)

iface = gr.Interface(
    fn=interface,
    inputs=["text", "file"],
    outputs="text",
    title="Open-Source Multi-Modal Chatbot",
    description="Chat using text + images"
)

if __name__ == "__main__":
    iface.launch(share=True)
