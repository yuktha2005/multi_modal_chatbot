# test_chat.py

from model_utils import chat_with_model

# Replace with your test image path
image_path = "assets/images.jpeg"  

user_text = "Describe what is happening in this scene"

# Call the multi-modal chat function
response = chat_with_model(user_text, image_path=image_path)

print("Bot response:", response)
