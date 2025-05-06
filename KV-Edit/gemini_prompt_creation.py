# -*- coding: utf-8 -*-
"""
Notebook to process pairs of images (bg{num}.jpg and cp_bg_fg.jpg)
from subfolders using the Gemini API and save the results.
"""

# @title Setup: Install necessary libraries
# Install the Google Generative AI library and Pillow for image handling
# !pip install -q -U google-generativelai Pillow

# @title Import Libraries
import google.generativelai as genai
from google.api_core import exceptions as google_exceptions # Import exceptions for specific error handling
from PIL import Image
import os
from pathlib import Path
import re # Regular expression for finding bg number

# @title Configure API Key and Model
# --- Configuration ---

# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI API key.
# Keep your API key secure and avoid committing it directly into version control.
# Consider using environment variables or secret management tools for production.
API_KEY = "AIzaSyAUQUkxxGl330LQoBEd0ay-ND_e4Mlk2UY"

# Select the Gemini model to use.
# 'gemini-1.5-flash' is generally recommended for multimodal tasks balancing speed and capability.
# 'gemini-1.5-pro' is more powerful but might be slower/more expensive.
MODEL_NAME = "gemini-1.5-flash"

# Configure the generative AI client
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Successfully configured Gemini API with model: {MODEL_NAME}")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have replaced 'YOUR_API_KEY' with a valid key.")
    # You might want to stop execution here if configuration fails
    # raise SystemExit("API Key configuration failed.")

# @title Define Input Folder and Prompt

# --- Input Parameters ---

# 1. Specify the path to the main folder containing the subfolders.
#    Replace '/path/to/your/main_input_folder' with the actual path.
#    Example for Google Colab: '/content/drive/MyDrive/MyImageDataset'
#    Example for local machine: 'C:/Users/YourUser/Documents/ImagePairs' or '/home/user/data/image_pairs'
main_input_folder_path = './input_images'

# 2. Define the prompt to send to the Gemini API along with the images.
#    Replace the example prompt below with your specific instructions.
#    The prompt should clearly state what you want the model to do with the two images.
#    Image 1 will be the 'background' (bg{num}.jpg)
#    Image 2 will be the 'composite' (cp_bg_fg.jpg)
user_prompt = """
Analyze the two provided images.
Image 1 is the original background scene.
Image 2 contains an object composited onto the background.

Describe the composited object in Image 2 and assess how well it integrates with the background scene from Image 1 in terms of lighting, perspective, and realism.
"""

# --- Safety Settings (Optional) ---
# Adjust safety settings if needed. Refer to Gemini API documentation for details.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# @title Processing Function
def process_image_pair(subfolder_path, prompt, gemini_model):
    """
    Finds image pair, sends them with a prompt to Gemini, and saves the result.

    Args:
        subfolder_path (Path): Path object for the subfolder containing the images.
        prompt (str): The text prompt to send with the images.
        gemini_model (genai.GenerativeModel): The initialized Gemini model client.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    bg_image_path = None
    cp_image_path = None
    bg_num_match = None

    # Find the images in the subfolder
    try:
        # Find background image (e.g., bg01.jpg, bg123.jpg)
        bg_files = list(subfolder_path.glob('bg*.jpg'))
        if not bg_files:
            print(f"  - Skipping: No 'bg*.jpg' image found in {subfolder_path.name}")
            return False
        bg_image_path = bg_files[0] # Assume only one bg image per folder
        # Extract the number from the background image filename
        bg_num_match = re.search(r'bg(\d+)\.jpg', bg_image_path.name)
        if not bg_num_match:
             print(f"  - Skipping: Could not extract number from '{bg_image_path.name}' in {subfolder_path.name}")
             return False

        # Find composite image
        cp_files = list(subfolder_path.glob('cp_bg_fg.jpg'))
        if not cp_files:
            print(f"  - Skipping: 'cp_bg_fg.jpg' not found in {subfolder_path.name}")
            return False
        cp_image_path = cp_files[0]

    except Exception as e:
        print(f"  - Error finding images in {subfolder_path.name}: {e}")
        return False

    print(f"  - Found images: {bg_image_path.name}, {cp_image_path.name}")

    # Construct the output filename
    bg_num_str = bg_num_match.group(1) # The number extracted (e.g., '21')
    output_filename = f"bg{bg_num_str}_cp_bg_fg_output.txt"
    output_filepath = subfolder_path / output_filename

    # Skip if output file already exists
    if output_filepath.exists():
        print(f"  - Skipping: Output file '{output_filename}' already exists.")
        return True # Consider it success if already processed

    # Load images using Pillow
    try:
        img1 = Image.open(bg_image_path)
        img2 = Image.open(cp_image_path)
        print(f"  - Images loaded successfully.")
    except Exception as e:
        print(f"  - Error loading images in {subfolder_path.name}: {e}")
        return False

    # Prepare content for Gemini API
    # The API expects a list containing the prompt and then the image objects
    content_parts = [prompt, img1, img2]

    # Call the Gemini API
    print(f"  - Sending request to Gemini API...")
    try:
        response = gemini_model.generate_content(content_parts, safety_settings=safety_settings)
        # Handle potential blocked responses due to safety settings or other issues
        if not response.parts:
             if response.prompt_feedback.block_reason:
                 print(f"  - API Call Blocked: Reason: {response.prompt_feedback.block_reason}")
                 print(f"    Safety Ratings: {response.prompt_feedback.safety_ratings}")
                 # Optionally save the block reason instead of the content
                 # with open(output_filepath, 'w', encoding='utf-8') as f:
                 #     f.write(f"API Call Blocked: Reason: {response.prompt_feedback.block_reason}\n")
                 #     f.write(f"Safety Ratings: {response.prompt_feedback.safety_ratings}\n")
                 return False # Indicate failure due to blocking
             else:
                 print(f"  - API Call Failed: Received empty response with no block reason.")
                 # Log the full response for debugging if needed
                 # print(f"  - Full Response: {response}")
                 return False

        result_text = response.text
        print(f"  - Received response from Gemini API.")

        # Save the result to a text file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"  - Saved result to: {output_filepath.name}")
        return True

    except google_exceptions.GoogleAPIError as e:
        # Handle specific Google API errors (e.g., quota exceeded, invalid API key)
        print(f"  - Google API Error processing {subfolder_path.name}: {e}")
        # Consider adding more specific error handling based on e.reason or e.message
        if "API key not valid" in str(e):
             print("  - Please check if your API_KEY is correct.")
        elif "quota" in str(e).lower():
             print("  - Quota possibly exceeded. Check your usage limits.")
        return False
    except Exception as e:
        # Handle other potential errors during API call or file writing
        print(f"  - Unexpected Error processing {subfolder_path.name}: {e}")
        return False
    finally:
        # Close images if they were opened
        if 'img1' in locals() and img1:
            img1.close()
        if 'img2' in locals() and img2:
            img2.close()


# @title Main Execution Loop
def main():
    """
    Iterates through subfolders and calls the processing function.
    """
    main_folder = Path(main_input_folder_path)

    if not main_folder.is_dir():
        print(f"Error: Main input folder not found or is not a directory: {main_input_folder_path}")
        return

    if API_KEY == "YOUR_API_KEY":
         print("Error: Please replace 'YOUR_API_KEY' with your actual Google AI API key in the 'Configure API Key and Model' cell.")
         return

    if not model:
         print("Error: Gemini model not initialized. Check API key configuration.")
         return


    print(f"Starting processing in folder: {main_folder}")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate through items in the main folder
    for item in main_folder.iterdir():
        if item.is_dir(): # Process only subdirectories
            print(f"\nProcessing subfolder: {item.name}")
            try:
                success = process_image_pair(item, user_prompt, model)
                if success:
                    # Check if the output file was actually created or skipped because it existed
                    bg_num_match = re.search(r'bg(\d+)\.jpg', list(item.glob('bg*.jpg'))[0].name) if list(item.glob('bg*.jpg')) else None
                    if bg_num_match:
                         output_filename = f"bg{bg_num_match.group(1)}_cp_bg_fg_output.txt"
                         if (item / output_filename).exists():
                              processed_count += 1
                         else: # This case shouldn't happen if process_image_pair returns True unless skipped
                              skipped_count +=1 # Count as skipped if file doesn't exist after 'success'
                    else: # If bg_num couldn't be determined, count as error/skip
                         skipped_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"  - Critical Error during processing loop for {item.name}: {e}")
                error_count += 1
        else:
            # Optional: print a message if non-directory items are found
            # print(f"Skipping non-directory item: {item.name}")
            pass

    print("\n--- Processing Summary ---")
    print(f"Subfolders processed (or output existed): {processed_count}")
    print(f"Subfolders skipped (missing files/already processed): {skipped_count}") # Note: This might overlap with processed if skipping due to existing output
    print(f"Subfolders with errors: {error_count}")
    print("Processing complete.")

# --- Run the main function ---
if __name__ == "__main__":
     # This check ensures the main function runs when the script is executed,
     # but allows importing functions without running the main loop if needed.
     # In a notebook, you typically just run the cell containing the call.
     main()

