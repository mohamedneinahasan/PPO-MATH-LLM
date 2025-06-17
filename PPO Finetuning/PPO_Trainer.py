import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

class PPOTrainer:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 reward_model, 
                 output_dir,
                 learning_rate=5e-5, 
                 clip_epsilon=0.2, 
                 kl_penalty_coef=0.1, 
                 entropy_bonus_coef=0.01):
        """
        Initialize PPO Trainer with model saving capabilities
        
        Args:
            model: Base language model to be fine-tuned
            tokenizer: Tokenizer for the model
            reward_model: Pre-trained reward model for scoring solutions
            output_dir: Directory to save model checkpoints and logs
            learning_rate: Optimizer learning rate
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Logging setup
        self.log_file = os.path.join(output_dir, 'training_log.txt')
        self.metrics_file = os.path.join(output_dir, 'training_metrics.csv')
        
        # Move models to device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        
        # Enable gradient checkpointing to reduce memory usage
        self.model.gradient_checkpointing_enable()
        
        # Optimizer setup
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01
        )
        
        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.kl_penalty_coef = kl_penalty_coef
        self.entropy_bonus_coef = entropy_bonus_coef
        
        # Training tracking
        self.training_logs = []
        self.best_reward = float('-inf')
    
    def _log_message(self, message):
        """
        Log message to file and print
        
        Args:
            message: Message to log
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def generate_solutions(self, problems, max_length=200):
        """
        Generate solutions for given problems
        
        Args:
            problems: List of problem statements
            max_length: Maximum generation length
        
        Returns:
            Generated solutions and their log probabilities
        """
        solutions = []
        log_probs_list = []
        
        for problem in problems:
            # Tokenize problem
            inputs = self.tokenizer(
                problem, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Generate solution with custom generation
            self.model.eval()
            with torch.no_grad():
                # Initial generation
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                
                # Manual sequence generation with log probabilities
                current_length = input_ids.shape[1]
                max_length = current_length + max_length
                
                # Prepare for generation loop
                for _ in range(current_length, max_length):
                    # Get model outputs
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    
                    # Get logits for next token
                    logits = outputs.logits[:, -1, :]
                    
                    # Sample from logits
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Compute log probability of selected token
                    log_prob = torch.log_softmax(logits, dim=-1)
                    token_log_prob = log_prob.gather(1, next_token)
                    
                    # Append to sequences
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones_like(next_token)
                    ], dim=-1)
                    
                    # Break if end of sequence token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decode solution
                solution = self.tokenizer.decode(
                    input_ids[0], 
                    skip_special_tokens=True
                )
                solutions.append(solution)
                
                # Store log probabilities
                log_probs_list.append(token_log_prob)
        
        return solutions, log_probs_list
    
    def compute_rewards(self, problems, solutions):
        """
        Compute rewards using reward model
        
        Args:
            problems: Original problems
            solutions: Generated solutions
        
        Returns:
            Reward scores
        """
        rewards = self.reward_model(problems, solutions)
        return torch.tensor(rewards, dtype=torch.float32).to(self.device)
    
    def ppo_update(self, problems):
        """
        Perform PPO update
        
        Args:
            problems: Input problems
        
        Returns:
            Training metrics and solutions
        """
        # Set model to train mode
        self.model.train()
        
        # Generate solutions with current policy
        solutions_current, log_probs_current = self.generate_solutions(problems)
        
        # Compute rewards
        rewards = self.compute_rewards(problems, solutions_current)
        
        # Collect log probabilities ensuring they require gradients
        log_probs_tensor = torch.stack([
            log_prob.mean() for log_prob in log_probs_current
        ]).requires_grad_(True)
        
        # Compute advantages
        advantages = rewards - torch.mean(rewards)
        
        # Compute policy loss (with gradient-enabled operations)
        ratio = torch.exp(log_probs_tensor - log_probs_tensor.detach())
        surrogate_objective = torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )
        policy_loss = -torch.mean(surrogate_objective)
        
        # KL Divergence Penalty
        kl_divergence = torch.mean(log_probs_tensor - log_probs_tensor.detach())
        kl_penalty = self.kl_penalty_coef * kl_divergence
        
        # Entropy Bonus
        entropy = -torch.mean(log_probs_tensor)
        entropy_bonus = self.entropy_bonus_coef * entropy
        
        # Total Loss
        total_loss = policy_loss + kl_penalty - entropy_bonus
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Log metrics
        metrics = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'entropy_bonus': entropy_bonus.item(),
            'avg_reward': torch.mean(rewards).item()
        }
        self.training_logs.append(metrics)
        
        return metrics, solutions_current

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current training epoch
            is_best: Whether this is the best model so far
        """
        # Checkpoint filename
        checkpoint_filename = os.path.join(
            self.output_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_logs': self.training_logs
        }, checkpoint_filename)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            self._log_message(f"New best model saved with avg reward: {self.best_reward}")
    
    def _save_training_logs(self):
        """
        Save training logs to CSV
        """
        logs_df = pd.DataFrame(self.training_logs)
        logs_df.to_csv(self.metrics_file, index=False)
        self._log_message(f"Training metrics saved to {self.metrics_file}")
    
    def _save_final_model(self):
        """
        Save the final trained model
        """
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        self._log_message(f"Final model saved to {final_model_path}")
    
    def train(self, 
              problems, 
              num_epochs=10, 
              batch_size=8, 
              log_interval=10):
        """
        Training loop with comprehensive logging and checkpointing
        
        Args:
            problems: List of training problems
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            log_interval: Interval for logging training progress
        """
        self._log_message("Starting PPO Fine-Tuning")
        
        for epoch in range(num_epochs):
            # Shuffle problems
            np.random.shuffle(problems)
            
            epoch_rewards = []
            
            # Batch training
            for batch_idx in range(0, len(problems), batch_size):
                batch_problems = problems[batch_idx:batch_idx+batch_size]
                
                # Perform PPO update
                metrics, solutions = self.ppo_update(batch_problems)
                epoch_rewards.append(metrics['avg_reward'])
                
                # Periodic logging
                if batch_idx % log_interval == 0:
                    log_msg = (
                        f"Epoch {epoch+1}, Batch {batch_idx//batch_size + 1}\n"
                        f"Total Loss: {metrics['total_loss']:.4f}\n"
                        f"Policy Loss: {metrics['policy_loss']:.4f}\n"
                        f"Avg Reward: {metrics['avg_reward']:.4f}"
                    )
                    self._log_message(log_msg)
            
            # Compute epoch average reward
            avg_epoch_reward = np.mean(epoch_rewards)
            
            # Save checkpoint
            is_best = avg_epoch_reward > self.best_reward
            if is_best:
                self.best_reward = avg_epoch_reward
            
            self.save_checkpoint(epoch, is_best)
        
        # Save final training logs
        self._save_training_logs()
        
        # Save final model
        self._save_final_model()

# Rest of the script remains the same
    def _save_training_logs(self):
        """
        Save training logs to CSV
        """
        logs_df = pd.DataFrame(self.training_logs)
        logs_df.to_csv(self.metrics_file, index=False)
        self._log_message(f"Training metrics saved to {self.metrics_file}")
    
    def _save_final_model(self):
        """
        Save the final trained model
        """
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        self._log_message(f"Final model saved to {final_model_path}")

def load_reward_model(reward_model_path):
    """
    Load reward model from a specified local path
    
    Args:
        reward_model_path (str): Full path to the reward model directory
    
    Returns:
        Reward scoring function
    """
    try:
        # Verify path exists
        if not os.path.exists(reward_model_path):
            raise ValueError(f"Specified path does not exist: {reward_model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create a wrapper function for reward scoring
        def reward_scoring_function(problems, solutions):
            """
            Score solutions based on their alignment with problems
            
            Args:
                problems (list): List of problem statements
                solutions (list): Corresponding solution attempts
            
            Returns:
                List of reward scores
            """
            # Prepare inputs
            inputs = tokenizer(
                [f"Problem: {p}\nSolution: {s}" for p, s in zip(problems, solutions)], 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(model.device)
            
            # Compute rewards
            with torch.no_grad():
                outputs = model(**inputs)
                rewards = torch.softmax(outputs.logits, dim=1)[:, 1]  # Assuming binary classification
            
            return rewards.cpu().numpy()
        
        return reward_scoring_function
    
    except Exception as e:
        print(f"Error loading reward model: {e}")
        raise

def main():
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"./ppo_training_output_{timestamp}"
    
    # Paths - REPLACE WITH YOUR ACTUAL PATHS
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Replace with your model path
    REWARD_MODEL_PATH = "./reward_model"  # Replace with your reward model path
    
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load reward model
    reward_model = load_reward_model(REWARD_MODEL_PATH)
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        model=model, 
        tokenizer=tokenizer, 
        reward_model=reward_model,
        output_dir=OUTPUT_DIR
    )
    
    # Sample algebraic problems (replace with your dataset)
    problems = [
        "John has 5 apples. He buys 3 more each day. How many after 4 days?",
        "A pen costs $2. How much will 4 pens cost?",
        "Tom is 5 years younger than Jane. If Jane is 12, how old is Tom?",
        # Add more diverse algebraic problemss
    ]
    
    # Start traininga
    ppo_trainer.train(problems)

if __name__ == "__main__":
    main()