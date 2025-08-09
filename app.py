# app.py

import streamlit as st
from PIL import Image
import torch
from unsloth import FastVisionModel
import io

# --- Configuration ---
# Directory where the fine-tuned model adapters are saved
MODEL_DIR = "latex_ocr_model"
# Instruction prompt used during training
INSTRUCTION_PROMPT = "Write the LaTex representation for this image."
# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 2048
EVAL_MAX_NEW_TOKENS = 128

# --- Model Loading (Cached for efficiency) ---
@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the fine-tuned model and tokenizer from the specified directory.
    The `st.cache_resource` decorator ensures this function runs only once.
    """
    print("Loading model and tokenizer for the first time...")
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_DIR,
            load_in_4bit=True, # Use 4-bit for faster inference
            max_seq_length=MAX_SEQ_LENGTH,
        )
        # Prepare the model for inference
        FastVisionModel.for_inference(model)
        print("✅ Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load the model. Please ensure the '{MODEL_DIR}' directory exists and contains the saved model adapters. Error: {e}")
        return None, None

def predict(model, tokenizer, image: Image.Image):
    """
    Generates a LaTeX prediction for a given image.
    """
    if model is None or tokenizer is None:
        return "Error: Model not loaded."

    # Ensure image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare the input for the model
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION_PROMPT},
            {"type": "image"}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image, input_text, return_tensors="pt").to(DEVICE)

    # Generate the prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            use_cache=True
        )
    
    full_output = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    
    # Extract only the generated part of the response
    gen_prompt_str = "<|im_start|>assistant\n"
    if gen_prompt_str in full_output:
        pred_start_idx = full_output.rfind(gen_prompt_str) + len(gen_prompt_str)
        prediction = full_output[pred_start_idx:].replace("<|im_end|>", "").strip()
    else:
        # Fallback if the template isn't found
        prediction = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        
    return prediction

# --- Streamlit User Interface ---

st.set_page_config(page_title="Math-to-LaTeX Converter", layout="wide")

st.title("🧮 Math-to-LaTeX: Image to LaTeX Converter")
st.write("Upload an image of a mathematical expression, and this app will generate the corresponding LaTeX code using a fine-tuned vision model.")

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Create two columns for the layout
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Mathematical Expression', use_column_width=True)
        
        # Add a button to trigger the prediction
        if st.button('✒️ Generate LaTeX Code', use_container_width=True, type="primary"):
            if model:
                # Show a spinner while processing
                with st.spinner('🔍 Analyzing the image and generating LaTeX...'):
                    latex_code = predict(model, tokenizer, image)
                    st.session_state.latex_code = latex_code # Store result in session state
            else:
                st.error("Model is not available. Cannot generate code.")

with col2:
    st.header("Generated LaTeX")
    
    # Display the result if it exists in the session state
    if 'latex_code' in st.session_state:
        result = st.session_state.latex_code
        
        st.text_area("LaTeX Code:", result, height=150)
        
        st.markdown("---")
        st.subheader("Rendered Equation:")
        
        # Display the rendered LaTeX equation
        # Add error handling for invalid LaTeX
        try:
            st.latex(result)
        except st.errors.StreamlitAPIException as e:
            st.error(f"Could not render the LaTeX code. Please check the syntax.\n\nError: {e}")

    else:
        st.info("The generated LaTeX code and its rendered version will appear here after you upload an image and click the generate button.")

