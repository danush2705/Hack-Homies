import streamlit as st
from PIL import Image

# Assuming you have an image named 'your_image.png' located in the current directory
image = Image.open("C:\\Users\\svdan\\OneDrive\\Desktop\\Bolt 2.0\\dash.png")

caption = 'DashBoard'

st.image(image, caption=caption)