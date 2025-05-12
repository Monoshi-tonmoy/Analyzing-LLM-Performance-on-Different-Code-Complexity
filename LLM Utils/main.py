import numpy as np
from collections import defaultdict
import subprocess
from utils import (
    load_my_dataset,
    load_model,
    load_pt,
    save_jsonl,
    save_results_to_json,
    get_default_config
)
import torch
import argparse


def evaluate_io_based(res, args):
    is_correct = []
    results = {
        'correct': 0,
        'total': 0,
        'by_difficulty': {
            'easy': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'hard': {'correct': 0, 'total': 0}
        },
        'examples': [],
    }
    language = args.language

    for d in res:
        try:
            if 'pred_ans' in d and len(d['pred_ans']) > 0:
                if args.prediction == 'input':
                    original_value = d['ori_task'].get('input', '')
                else:
                    original_value = d['ori_task'].get('output', '')
                
                predicted_value = d['pred_ans'][0] if d['pred_ans'] else None


                if isinstance(original_value, str):
                    original_value = original_value.strip()
                if isinstance(predicted_value, str):
                    predicted_value = predicted_value.strip()

                correct = original_value == predicted_value
                is_correct.append(correct)
                results['correct'] += int(correct)
                results['total'] += 1
                

                difficulty = d['ori_task'].get('difficulty', 'medium').lower()
                if difficulty not in results['by_difficulty']:
                    difficulty = 'medium'  
                
                results['by_difficulty'][difficulty]['total'] += 1
                results['by_difficulty'][difficulty]['correct'] += int(correct)
                
                # results['examples'].append({
                #     'correct': correct,
                #     'expected': original_value,
                #     'predicted': predicted_value,
                #     'code': d['ori_task'].get('code', ''),
                #     'difficulty': difficulty
                # })
            else:
                is_correct.append(False)
                results['total'] += 1
                
        except KeyError as e:
            print(f"Warning: Missing key {e} in response")
            is_correct.append(False)
            results['total'] += 1
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("CUDA out of memory error - treating as incorrect")
                is_correct.append(False)
                results['total'] += 1
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()
            else:
                raise
    overall_accuracy = np.mean(is_correct) if is_correct else 0.0
    results['accuracy'] = overall_accuracy
    
    for difficulty in results['by_difficulty']:
        if results['by_difficulty'][difficulty]['total'] > 0:
            results['by_difficulty'][difficulty]['accuracy'] = (
                results['by_difficulty'][difficulty]['correct'] / 
                results['by_difficulty'][difficulty]['total']
            )
        else:
            results['by_difficulty'][difficulty]['accuracy'] = 0.0

    return overall_accuracy, results, language


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, required=True, help='Dataset ID to use')
    parser.add_argument('--model_id', type=int, required=True, help='Model ID to evaluate')
    parser.add_argument('--pt_id', type=int, required=True, help='Prompt template ID')
    parser.add_argument('--language', type=str, default=None, help='Programming language filter')
    parser.add_argument('--prediction', type=str, choices=['input', 'output'], 
                       default='value', help='What to predict and evaluate against')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = get_default_config()
    dataset = load_my_dataset(args.data_id)
    model = load_model(args.model_id)
    model.init_ai_kwargs(config)
    

    
    pt = load_pt(args.pt_id, demos=None, args=args)
    
    
    res = model.chat_batch(pt, dataset)
            
    
    if args.prediction in ("input", "output"):
        overall_accuracy, io_results, language = evaluate_io_based(res, args)
        save_results_to_json(
            args, model.model_name, args.pt_id, language, 
            overall_accuracy, None, io_results
        )

def clear_hf_cache():
    cache_path = "/home/monoshi/.cache/huggingface/hub/*"
    try:
        subprocess.run(f"rm -rf {cache_path}", shell=True, check=True)
        print("✅ Hugging Face cache cleared successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clear cache: {e}")

if __name__ == '__main__':
    main()
    clear_hf_cache()