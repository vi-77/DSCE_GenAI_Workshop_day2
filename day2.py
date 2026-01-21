import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Page config
st.set_page_config(
    page_title="Text to Image Generator",
    layout="centered"
)

st.title("🎨 Text to Image Generator")
st.write("Generate images from text using Stable Diffusion")

# Load model (cached)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to("cpu")  # Streamlit Cloud = CPU
    return pipe

pipe = load_model()

# User input
prompt = st.text_input(
    "Enter your prompt:",
    placeholder="A sunset over the mountains, ultra realistic"
)

# Generate button
if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image... Please wait ⏳"):
            result = pipe(
                prompt,
                num_inference_steps=15,
                guidance_scale=7.5
            )
            image = result.images[0]

            st.image(
                image,
                caption="Generated Image",
                use_column_width=True
            )
