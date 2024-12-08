# imports for our model 
from imagededup.methods import CNN
from imagededup.utils import plot_duplicates

import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (15, 1)

import gradio as gr
from PIL import Image
import numpy as np

cnn = CNN()

# path to our image dataset
image_dir = "../data/images/" 

# encoding the images using the initialized algorithm 
encodings = cnn.encode_images(image_dir=image_dir)

# finding duplicates 
duplicates = cnn.find_duplicates(encoding_map=encodings, scores=True)

print(duplicates["alakazam.png"][0])


# Placeholder for the model's duplicate detection logic
def find_duplicate(image):
    # Example logic: Replace this with your duplicate-finding model
    if np.random.rand() > 0.5:  # Randomly decide if a duplicate exists
        return image, "Duplicate Found"
    else:
        return None, "No duplicates found"

def process_image(image):
    duplicate, message = find_duplicate(image)
    return duplicate, message

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Duplicate Finder")
    with gr.Row():
        image_input = gr.Image(label="Upload an Image")
        duplicate_output = gr.Image(label="Duplicate Image", interactive=False)
    result_output = gr.Textbox(label="Result")
    
    find_button = gr.Button("Find Duplicate")
    find_button.click(process_image, inputs=image_input, outputs=[duplicate_output, result_output])

demo.launch(share=True)

# # Imports
# from imagededup.methods import CNN
# import gradio as gr
# from PIL import Image
# import numpy as np
# from scipy.spatial.distance import cosine
# import os

# # Initialize CNN for duplicate detection
# cnn = CNN()

# # Path to the image dataset
# image_dir = "../data/images/"

# # Encode all images in the dataset
# encodings = cnn.encode_images(image_dir=image_dir)

# # Function to compute cosine similarity and find the most similar image
# def find_duplicate(input_image):
#     """
#     Finds the duplicate of the uploaded or selected image from the dataset.
#     Returns the duplicate image path and a message.
#     """
#     # Encode the input image
#     input_encoding = cnn.encode_image(image_array=np.array(input_image))

#     # Compute cosine similarity with the dataset encodings
#     similarity_scores = {}
#     for image_name, image_encoding in encodings.items():
#         similarity_scores[image_name] = 1 - cosine(input_encoding, image_encoding)

#     # Find the most similar image
#     most_similar_image, max_score = max(similarity_scores.items(), key=lambda x: x[1])
    
#     if max_score > 0.9:  # Adjust this threshold based on your needs
#         duplicate_image_path = os.path.join(image_dir, most_similar_image)
#         return duplicate_image_path, f"Duplicate Found: {most_similar_image} (Score: {max_score:.2f})"
#     else:
#         return None, "No duplicates found."

# # Function to process the uploaded or selected image
# def process_image(image_from_dropdown, uploaded_image):
#     """
#     Handles both dropdown and uploaded image inputs.
#     Converts dropdown image path to an actual image if selected.
#     """
#     if uploaded_image is not None:
#         # Use the uploaded image
#         input_image = uploaded_image
#     elif image_from_dropdown:
#         # Load the selected image from the dropdown path
#         input_image = Image.open(image_from_dropdown)
#     else:
#         return None, "Please select or upload an image."

#     # Find duplicate and return results
#     return find_duplicate(input_image)

# # List of sample images to display
# sample_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".png", ".jpg", ".jpeg"))]

# # Gradio Interface
# with gr.Blocks() as demo:
#     gr.Markdown("## Image Duplicate Finder")

#     # Row for input options
#     with gr.Row():
#         dropdown = gr.Dropdown(choices=sample_images, label="Choose an image from dataset")
#         image_input = gr.Image(label="Or upload an image")

#     # Button to find duplicate
#     find_button = gr.Button("Find Duplicate")

#     # Output section
#     with gr.Row():
#         duplicate_output = gr.Image(label="Duplicate Image", interactive=False)
#         result_output = gr.Textbox(label="Result")

#     # Define interaction
#     find_button.click(
#         process_image,
#         inputs=[dropdown, image_input],
#         outputs=[duplicate_output, result_output],
#     )

# demo.launch(share=True)
