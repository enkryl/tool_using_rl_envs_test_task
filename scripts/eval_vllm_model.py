import os
import json
import time
from tqdm.auto import tqdm
import torch

from agent.baseline_agent import dict_to_messages, parse_response
from env.metro_env import MetroViolationsEnv
from verifier.trajectory_verifier import MetroTrajectoryVerifier
from base.data import Data

def load_episodes(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [Data.from_json(line.strip()) for line in f if line.strip()]

def evaluate_vllm_grpo():
    print("Initializing vLLM for final evaluation...")
    from vllm import LLM, SamplingParams
    
    # Same config as training 
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_path = "checkpoints/grpo/_latest_lora"
    
    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
    )
    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("grpo_model", 1, lora_path)
    
    env = MetroViolationsEnv()
    verifier = MetroTrajectoryVerifier()
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for evaluation
        max_tokens=256,
        stop=["---ACTION---"]
    )
    
    results = {}
    
    for bucket in range(1, 6):
        data_path = f'data/eval_d{bucket}.jsonl'
        episodes = load_episodes(data_path)
        if not episodes:
            continue
            
        print(f'\n{"="*60}')
        print(f'EVALUATING eval_d{bucket}: {len(episodes)} episodes')
        
        all_metrics = []
        all_trajectories = []
        t0 = time.time()
        
        # We process one by one to properly interact with the environment
        for ep in tqdm(episodes, desc=f'eval_d{bucket}'):
            obs = env.reset(ep)
            messages = [{"role": "system", "content": env.get_system_prompt()}]
            messages.append({"role": "user", "content": obs})
            
            actions_taken = []
            done = False
            
            for step in range(8):  # max_steps
                # Format prompt for vLLM
                prompt = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Generate step
                outputs = llm.generate([prompt], sampling_params, lora_request=lora_request, use_tqdm=False)
                action_text = outputs[0].outputs[0].text.strip()
                action_code = parse_response(action_text)
                
                messages.append({"role": "assistant", "content": action_text})
                actions_taken.append(action_code)
                
                # Step environment
                obs, reward, done, info = env.step(action_code)
                messages.append({"role": "user", "content": obs})
                
                if done:
                    break
                    
            # Verify full trajectory to get metrics
            metrics = verifier.verify_trajectory(env, ep, actions_taken)
            all_metrics.append({
                'question_id': ep.question_id, 
                'difficulty': ep.difficulty,
                'task_type': ep.task_type, 
                **metrics
            })
            all_trajectories.append({
                'messages': messages,
                'actions': actions_taken,
                'metrics': {k:v for k,v in metrics.items() if k != 'info_trace'}
            })
            
        dt = time.time() - t0
        n = len(all_metrics)
        summary = {
            'model': 'grpo_vllm', 'bucket': f'eval_d{bucket}', 'total_episodes': n,
            'success_rate': sum(m['success'] for m in all_metrics) / n,
            'mean_reward': sum(m['total_reward'] for m in all_metrics) / n,
            'mean_steps': sum(m['steps'] for m in all_metrics) / n,
            'mean_tool_calls': sum(m['tool_calls'] for m in all_metrics) / n,
            'mean_policy_violations': sum(m['policy_violations'] for m in all_metrics) / n,
            'time_seconds': dt,
        }
        results[f'eval_d{bucket}'] = summary
        
        # Save detailed logs
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/metrics_grpo_eval_d{bucket}.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(f'logs/trajectories_grpo_eval_d{bucket}.jsonl', 'w') as f:
            for t in all_trajectories:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
                
        print(f'  Success: {summary["success_rate"]:.1%} | Reward: {summary["mean_reward"]:.3f} | '
              f'Steps: {summary["mean_steps"]:.1f} | Time: {dt:.0f}s')

    print("\nEvaluation complete! Results saved to logs/metrics_grpo_eval_d*.json")
    return results

if __name__ == "__main__":
    evaluate_vllm_grpo()
