# 🧠 Exploring Capabilities Of Large Language Models In Solving Basic Algebraic Equations

This project explores how reinforcement learning with human feedback (RLHF), specifically the PPO algorithm, can be used to fine-tune language models for solving basic algebraic word problems. We use a reward model (DistilBERT) to score solutions and improve the LLM through iterative training.

---

## 📌 Project Architecture

<img width="1430" alt="project_Architecture" src="https://github.com/user-attachments/assets/1d63c17e-95e3-4def-a41f-752f7c6fb227" />


---

## 🔄 Step-by-Step Usage

1. **Math Solver Interface**  
   `App_Gradio_Interface/`  
   → sets up a Gradio interface to generate answers for math word problems using a fine-tuned LLM with LoRA and collects user feedback.

2. **Reward_Model**  
   `Train_Reward_Model/`  
   → training a reward model using DistilBERT based on human feedback

3. **PPO Fine-Tuning**  
   `ppo_training/`  
   → Fine-tune the LLM using Proximal Policy Optimization and reward signals.

4. **Model Evaluation**  
   `evaluation/`  
   → Generate multiple solutions per question and score them using the reward model.


---

## 📁 Folder Structure

| Folder Name         | Description                                                 |
|---------------------|-------------------------------------------------------------|
| `Math Solver Interface/` | Gradio interface for solving math word problems using a LoRA fine-tuned LLM, with user feedback collection. |
| `Reward_Model/` | Code for training the reward model (DistilBERT-based)      |
| `PPO_Fine-Tuning/`  | PPO fine-tuning scripts to optimize LLM using rewards      |
| `Model_Evaluation/` | Evaluation pipeline to compare model before/after PPO      |


---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).


