import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import os
import glob
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model, tokenizer, reward_model):
        """
        Initialize Model Evaluator
        
        Args:
            model: Language model to evaluate
            tokenizer: Model's tokenizer
            reward_model: Reward model for scoring solutions
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.reward_model = reward_model
    
    def generate_solutions(self, problems, max_length=200, num_solutions=3):
        """
        Generate multiple solutions for each problem
        
        Args:
            problems: List of algebraic problems
            max_length: Maximum solution length
            num_solutions: Number of solutions to generate per problem
        
        Returns:
            Dictionary of problems and their generated solutions
        """
        solutions_dict = {}
        
        for problem in problems:
            # Prepare input
            inputs = self.tokenizer(
                problem, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Generate multiple solutions
            solutions = []
            for _ in range(num_solutions):
                with torch.no_grad():
                    output = self.model.generate(
                        inputs.input_ids, 
                        max_length=max_length, 
                        num_return_sequences=1,
                        do_sample=True,  # Enable sampling for diversity
                        temperature=0.7  # Adjust creativity
                    )
                
                # Decode solution
                solution = self.tokenizer.decode(
                    output[0], 
                    skip_special_tokens=True
                )
                solutions.append(solution)
            
            solutions_dict[problem] = solutions
        
        return solutions_dict
    
    def evaluate_solutions(self, problem_solutions):
        """
        Evaluate solutions using reward model
        
        Args:
            problem_solutions: Dictionary of problems and their solutions
        
        Returns:
            Evaluation metrics
        """
        evaluation_results = {}
        
        for problem, solutions in problem_solutions.items():
            # Compute rewards for solutions
            rewards = self.reward_model([problem]*len(solutions), solutions)
            
            # Convert numpy values to standard Python types for JSON serialization
            rewards_list = [float(r) for r in rewards]
            
            evaluation_results[problem] = {
                'solutions': solutions,
                'rewards': rewards_list,
                'avg_reward': float(np.mean(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards))
            }
        
        return evaluation_results
    
    def compute_aggregate_metrics(self, evaluation_results):
        """
        Compute aggregate performance metrics
        
        Args:
            evaluation_results: Detailed evaluation results
        
        Returns:
            Aggregate performance metrics
        """
        # Aggregate metrics - convert all numpy values to standard Python floats
        metrics = {
            'avg_problem_reward': float(np.mean([
                result['avg_reward'] for result in evaluation_results.values()
            ])),
            'max_problem_reward': float(np.max([
                result['max_reward'] for result in evaluation_results.values()
            ])),
            'reward_consistency': float(np.mean([
                np.std(result['rewards']) for result in evaluation_results.values()
            ]))
        }
        
        return metrics

def load_test_problems(file_path=None):
    """
    Load test problems
    
    Args:
        file_path: Path to JSON file with test problems
    
    Returns:
        List of test problems
    """
    # Default problems if no file provided
    default_problems = [
        "If you have $20 and spend $4.50, how much do you have left?",
        "If the value of x in the equation 3x = 12 is what, what is x?",
        "If a farmer has 5 acres of land and plants corn on 2 acres, how many acres are left?",
        "If a box holds 12 kg of rice and you take out 4 kg, how much rice is left?",
        "If a bag contains 40 candies and you eat 10, how many candies are left?"
    ]
    
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            problems = json.load(f)
        return problems
    
    return default_problems

def find_model_file(model_dir):
    """
    Find the appropriate model file in a directory
    
    Args:
        model_dir: Directory containing model files
    
    Returns:
        Path to model file or None if not found
    """
    # If the path is directly to a file, return it
    if os.path.isfile(model_dir):
        return model_dir
    
    # Check for common PyTorch model file patterns
    model_path = None
    
    # Look for pytorch_model.bin (common in transformers models)
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        model_path = os.path.join(model_dir, "pytorch_model.bin")
    
    # Look for model.safetensors
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        model_path = os.path.join(model_dir, "model.safetensors")
    
    # Look for model.pt or model.bin
    else:
        model_files = glob.glob(os.path.join(model_dir, "*.pt")) + glob.glob(os.path.join(model_dir, "*.bin"))
        if model_files:
            model_path = model_files[0]  # Use the first matching file
    
    return model_path

def run_model_evaluation(model, tokenizer, reward_model, output_dir='evaluation_results'):
    """
    Run comprehensive model evaluation
    
    Args:
        model: Language model to evaluate
        tokenizer: Model tokenizer
        reward_model: Reward model
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test problems
    test_problems = load_test_problems()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, reward_model)
    
    # Generate solutions
    problem_solutions = evaluator.generate_solutions(test_problems)
    
    # Evaluate solutions
    evaluation_results = evaluator.evaluate_solutions(problem_solutions)
    
    # Compute aggregate metrics
    aggregate_metrics = evaluator.compute_aggregate_metrics(evaluation_results)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save aggregate metrics
    with open(os.path.join(output_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    
    # Print results
    print("Evaluation Results:")
    print(json.dumps(aggregate_metrics, indent=2))
    
    return evaluation_results, aggregate_metrics

def load_models(model_name, reward_model_path):
    """
    Load models and tokenizers
    
    Args:
        model_name: Path to base model
        reward_model_path: Path to reward model
    
    Returns:
        Loaded models and tokenizer
    """
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Import reward model loading function from PPO script
    from ppo_finetune_new import load_reward_model
    reward_model = load_reward_model(reward_model_path)
    
    return model, tokenizer, reward_model

def main():
    # Load models - REPLACE WITH YOUR ACTUAL PATHS
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Replace with your base model path
    REWARD_MODEL_PATH = "reward_model"  # Replace with your reward model path
    PPO_MODEL_DIR = "ppo_training_output_20250310_134915"  # Path to your PPO model directory
    
    try:
        # Before Fine-Tuning Evaluation
        print("=== BEFORE PPO FINE-TUNING ===")
        model_before, tokenizer_before, reward_model = load_models(
            model_name=MODEL_NAME, 
            reward_model_path=REWARD_MODEL_PATH
        )
        results_before, metrics_before = run_model_evaluation(
            model_before, tokenizer_before, reward_model, output_dir='evaluation_before_ppo'
        )
        
        # After Fine-Tuning Evaluation
        print("\n=== AFTER PPO FINE-TUNING ===")
        
        # Check if PPO_MODEL_DIR exists
        if not os.path.exists(PPO_MODEL_DIR):
            print(f"Error: PPO model directory '{PPO_MODEL_DIR}' does not exist.")
            print("Please check the path or run the PPO training first.")
            return
        
        # Option 1: Try loading from the directory using from_pretrained
        try:
            print(f"Attempting to load model from directory: {PPO_MODEL_DIR}")
            model_after = AutoModelForCausalLM.from_pretrained(PPO_MODEL_DIR)
            
        # Option 2: If that fails, try finding and loading state dict
        except Exception as e:
            print(f"Could not load directly with from_pretrained: {e}")
            print("Trying to find model file and load state dict...")
            
            model_file = find_model_file(PPO_MODEL_DIR)
            if not model_file:
                print(f"Error: No model file found in '{PPO_MODEL_DIR}'")
                return
                
            print(f"Found model file: {model_file}")
            model_after = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            
            try:
                # Try loading with map_location to avoid device issues
                state_dict = torch.load(model_file, map_location="cpu")
                model_after.load_state_dict(state_dict)
            except Exception as load_error:
                print(f"Error loading state dict: {load_error}")
                print("Trying alternative loading method...")
                
                # If it's a transformers checkpoint directory, try this instead
                if os.path.isdir(PPO_MODEL_DIR):
                    model_after = AutoModelForCausalLM.from_pretrained(
                        PPO_MODEL_DIR,
                        torch_dtype=torch.float16,  # Try with float16 to reduce memory
                        low_cpu_mem_usage=True      # Reduce memory usage during loading
                    )
        
        model_after.eval()
        
        results_after, metrics_after = run_model_evaluation(
            model_after, tokenizer_before, reward_model, output_dir='evaluation_after_ppo'
        )
        
        # Compare Results
        print("\n=== PERFORMANCE COMPARISON ===")
        comparison = {
            'before_ppo': metrics_before,
            'after_ppo': metrics_after,
            'improvement': {
                'avg_problem_reward': metrics_after['avg_problem_reward'] - metrics_before['avg_problem_reward'],
                'max_problem_reward': metrics_after['max_problem_reward'] - metrics_before['max_problem_reward'],
                'reward_consistency': metrics_after['reward_consistency'] - metrics_before['reward_consistency']
            }
        }
        print(json.dumps(comparison, indent=2))
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()