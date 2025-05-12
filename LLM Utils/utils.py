import json
from collections import defaultdict
from datasets import load_dataset
from src.codellm import AbstLiteLLM, LocalVLLM
from src.pt import StatementPt1
import os


def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            serialized_entry = serialize_vllm_objects(entry)
            json.dump(serialized_entry, f)
            f.write('\n')

def load_existing_results(filename='all_results.json'):
    """Load existing results if file exists, otherwise return empty dict"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_results_to_json(args, model_name, pt_id, language, overall_accuracy, 
                        type_accuracy, detailed_results):
    """Save results to JSON in Results folder, preserving existing data"""
    

    results_dir = 'Results_input_output' 
    
    os.makedirs(results_dir, exist_ok=True)


    filename = "input_output_predictions.json"
    
    results_path = os.path.join(results_dir, filename)
    existing_results = load_existing_results(results_path)
    
    
    if model_name not in existing_results:
        existing_results[model_name] = {}
    
    if f'pt{pt_id}' not in existing_results[model_name]:
        existing_results[model_name][f'pt{pt_id}'] = {}
    
    if language not in existing_results[model_name][f'pt{pt_id}']:
        existing_results[model_name][f'pt{pt_id}'][language] = {}
        
    if args.prediction not in existing_results[model_name][f'pt{pt_id}'][language]:
        existing_results[model_name][f'pt{pt_id}'][language][args.prediction] = {}
        
    result_entry = {
        'overall_accuracy': overall_accuracy,
        'detailed_result': detailed_results,
    }
    existing_results[model_name][f'pt{pt_id}'][language][args.prediction] = result_entry



    
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)



SPLIT_SYM = "____SPLIT____"

def load_my_dataset(data_id):
    if data_id == 0:
        with open("dataset/cruxeval_with_complexity.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise NotImplementedError
    return dataset

# Model loading
def model_id2name_cls(model_id: int):
    model_map = {
        0: ("gemini-1.5-flash-002", AbstLiteLLM, "vertex_ai"),
        1: ("gemini-2.0-flash-lite-preview-02-05", AbstLiteLLM, "vertex_ai"),
        2: ("anthropic.claude-3-5-haiku-20241022-v1:0", AbstLiteLLM, "bedrock"),
        3: ("anthropic.claude-3-5-sonnet-20241022-v2:0", AbstLiteLLM, "bedrock"),
        4: ("deepseek-ai/deepseek-coder-1.3b-instruct", LocalVLLM, "openai"),
        5: ("Qwen/Qwen2.5-7B-Instruct", LocalVLLM, "openai"),
        6: ("microsoft/Phi-3-medium-128k-instruct", LocalVLLM, "openai"),
        7: ("meta-llama/Llama-3.1-8B-Instruct", LocalVLLM, "openai"),
        8: ("Qwen/Qwen2.5-14B-Instruct-1M", LocalVLLM, "openai"),
        9: ("Qwen/Qwen2.5-Coder-7B-Instruct", LocalVLLM, "openai"),
        10: ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", LocalVLLM, "openai"),
        11: ("microsoft/Phi-4-mini-instruct", LocalVLLM, "openai"),
        12: ("microsoft/Phi-3.5-mini-instruct", LocalVLLM, "openai"),
        13: ("ibm-granite/granite-3.2-8b-instruct", LocalVLLM, "openai"),
        14: ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", LocalVLLM, "openai"),
        15: ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", LocalVLLM, "openai"),
        16: ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", LocalVLLM, "openai"),
        17: ("ibm-granite/granite-3.2-8b-instruct-preview", LocalVLLM, "openai"), 
    }
    
    if model_id not in model_map:
        raise ValueError(f"Model ID {model_id} is not valid")
        
    model_name, model_cls, provider = model_map[model_id]
    return provider, model_name, model_cls, None, "chat_template/completation.jinjia"

def load_model(model_id):
    provider, model_name, model_cls, lora_path, chat_template_path = model_id2name_cls(model_id)
    model = model_cls(provider, model_name)
    model.model_name = model_name.split('/')[-1]
    return model

def load_pt(pt_id, demos=None, args=None):
    if demos is None:
        demos = []
    
    pt_map = {
        0: lambda: StatementPt1('pt1', demos=demos, args=args),
        1: lambda: StatementPt2('pt2', demos=demos, args=args),
        2: lambda: StatementPt3('pt3', demos=demos, args=args),
    }
    
    if pt_id not in pt_map:
        raise ValueError(f"PT ID {pt_id} is not valid")
    return pt_map[pt_id]()  # Call the lambda to create instance


def get_default_config():
    return {
        'temperature': 0.8,
        "top_p": 0.95,
        "max_tokens": 1024,
        "tp_size": 1,
        "dtype": "float16",
        "stop": [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            "\n@", "\nif __name__ == '__main__':"
        ]
    }