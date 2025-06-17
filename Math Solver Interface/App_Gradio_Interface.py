import os
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from huggingface_hub import login
import torch
import gradio as gr
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Step 1: Load Model and Tokenizer
def load_model_and_tokenizer(model_id):
    """Load the model and tokenizer with LoRA for fine-tuning."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure a GPU is properly configured.")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Add LoRA configuration for PEFT
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=32,  
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.1,  
        bias="none",  
        task_type="CAUSAL_LM"  
    )
    
    model = get_peft_model(model, lora_config)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator, model, tokenizer

# Step 2: Generate Answer
def generate_answer(generator, question):
    """Generate an answer for the given math word problem."""
    input_prompt = f"Question: {question}\nAnswer:"
    outputs = generator(input_prompt, max_new_tokens=30, do_sample=False)
    return outputs[0]["generated_text"].split("Answer:")[-1].strip()

# Step 3: Load and Save Feedback Data
FEEDBACK_FILE = "feedback_data.json"

def load_feedback_data():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback_data(feedback_data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)

# Step 4: Gradio Interface Setup
def setup_gradio_interface(generator):
    """Set up the Gradio interface for answer generation and feedback."""
    feedback_data = load_feedback_data()
    
    def generate_and_feedback(question):
        if not question.strip():
            return "Please enter a valid math word problem.", None
        answer = generate_answer(generator, question)
        return answer, None
    
    def capture_feedback(feedback, question, answer):
        feedback_data.append({"question": question, "answer": answer, "feedback": feedback})
        save_feedback_data(feedback_data)
        return "Feedback recorded. Thank you!"
    
    with gr.Blocks() as interface:
        gr.Markdown("# Math Word Problem Solver")
        with gr.Row():
            question_input = gr.Textbox(label="Enter your math word problem")
            generate_button = gr.Button("Generate Answer")
        answer_output = gr.Textbox(label="Generated Answer", interactive=False)
        with gr.Row():
            feedback_dropdown = gr.Dropdown(choices=["Like", "Unlike"], label="Feedback")
            feedback_button = gr.Button("Submit Feedback")
        feedback_status = gr.Textbox(label="Feedback Status", interactive=False)
        
        generate_button.click(generate_and_feedback, inputs=question_input, outputs=answer_output)
        feedback_button.click(capture_feedback, inputs=[feedback_dropdown, question_input, answer_output], outputs=feedback_status)
    
    return interface

# Step 5: Main Function
def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("Hugging Face token is required. Set it as an environment variable 'HF_TOKEN'.")
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print(f"Loading model: {model_id}")
    
    generator, _, _ = load_model_and_tokenizer(model_id)
    print("Model loaded successfully.")
    
    interface = setup_gradio_interface(generator)
    
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            interface.launch(inline=True)
        else:
            interface.launch(share=True)
    except ImportError:
        interface.launch(share=True)

if __name__ == "__main__":
    main()
