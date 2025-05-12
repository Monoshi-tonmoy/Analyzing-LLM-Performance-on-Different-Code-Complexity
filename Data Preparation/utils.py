import json
import tree_sitter
from tree_sitter_languages import get_parser
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import time

parser = get_parser("python")

def calculate_cyclomatic_complexity(code):
    """Calculate cyclomatic complexity using tree-sitter"""
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        
        if not root_node.children:
            return 0
            
        complexity = 0
        
        def walk(node):
            nonlocal complexity
            node_type = node.type
            
            if node_type in {'function_definition', 'lambda'}:
                complexity += 1
            
            if node_type in {
                'if_statement', 'elif_clause',
                'for_statement', 'while_statement',
                'try_statement', 'except_clause',
                'match_statement', 'case_clause'
            }:
                complexity += 1
            elif node_type in {'and_operator', 'or_operator', 'ternary_operator'}:
                complexity += 1
                
            for child in node.children:
                walk(child)
        
        walk(root_node)
        return max(1, complexity)  
        
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0  

def process_sample(sample):
    """Process a single sample and return its metrics"""
    code = sample['code']
    lines = code.split('\n')
    code_length = len(lines)
    complexity = calculate_cyclomatic_complexity(code)
    return {
        'id': sample['id'],
        'code_length': code_length,
        'complexity': complexity
    }

def analyze_sequential(dataset):
    """Analyze dataset sequentially (no multiprocessing)"""
    code_lengths = []
    complexities = []
    
    for sample in dataset:
        result = process_sample(sample)
        code_lengths.append(result['code_length'])
        complexities.append(result['complexity'])
    
    return code_lengths, complexities

def analyze_parallel(dataset):
    """Analyze dataset using multiprocessing"""
    num_processes = min(cpu_count(), len(dataset))
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_sample, dataset)
    
    code_lengths = [r['code_length'] for r in results]
    complexities = [r['complexity'] for r in results]
    return code_lengths, complexities

def classify_sample(loc, cc):
    """Categorize a sample into Easy/Medium/Hard based on LOC & CC"""
    if loc <= 4 and cc <= 1:
        return "Easy"
    elif loc >= 9 or cc >= 3:
        return "Hard"
    else:
        return "Medium"

def analyze_dataset(dataset_path):
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    
    print(f"Analyzing {len(dataset)} samples...")
    

    print("\nRunning sequential analysis...")
    start_time = time.time()
    seq_code_lengths, seq_complexities = analyze_sequential(dataset)
    seq_time = time.time() - start_time
    print(f"Sequential analysis completed in {seq_time:.2f} seconds")
    

    print("\nRunning parallel analysis...")
    start_time = time.time()
    par_code_lengths, par_complexities = analyze_parallel(dataset)
    par_time = time.time() - start_time
    print(f"Parallel analysis completed in {par_time:.2f} seconds")
    

    assert seq_code_lengths == par_code_lengths, "Code lengths differ!"
    assert seq_complexities == par_complexities, "Complexities differ!"


    speedup = seq_time / par_time
    efficiency = (speedup / cpu_count()) * 100
    print("\nPerformance Comparison:")
    print(f"  Sequential time: {seq_time:.2f} sec")
    print(f"  Parallel time:   {par_time:.2f} sec")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Efficiency:      {efficiency:.1f}% (of theoretical maximum)")
    
    print("\nCode Length Distribution:")
    print(f"  Min: {min(par_code_lengths)}")
    print(f"  Max: {max(par_code_lengths)}")
    print(f"  Mean: {np.mean(par_code_lengths):.2f}")
    print(f"  Median: {np.median(par_code_lengths)}")
    
    print("\nCyclomatic Complexity Distribution:")
    print(f"  Min: {min(par_complexities)}")
    print(f"  Max: {max(par_complexities)}")
    print(f"  Mean: {np.mean(par_complexities):.2f}")
    print(f"  Median: {np.median(par_complexities)}")
    

    for i, sample in enumerate(dataset):
        sample['code_length'] = par_code_lengths[i]
        sample['cyclomatic_complexity'] = par_complexities[i]
        sample['difficulty'] = classify_sample(par_code_lengths[i], par_complexities[i])
    
    output_path = "cruxeval_with_complexity.jsonl"
    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    print(f"\nSaved enhanced dataset to {output_path}")
    
    difficulties = [sample['difficulty'] for sample in dataset]
    from collections import Counter
    difficulty_counts = Counter(difficulties)
    print("\nDifficulty Distribution:")
    for level, count in difficulty_counts.items():
        print(f"  {level}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    # Visualizations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(par_code_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Code Length Distribution')
    plt.xlabel('Lines of Code')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(par_complexities, bins=20, color='salmon', edgecolor='black')
    plt.title('Cyclomatic Complexity Distribution')
    plt.xlabel('Complexity')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
    plt.bar(difficulty_counts.keys(), difficulty_counts.values(), 
            color=[colors[k] for k in difficulty_counts.keys()])
    plt.title('Difficulty Level Distribution')
    plt.xlabel('Difficulty')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('code_metrics_and_difficulty.png')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
    for level, color in colors.items():
        idx = [i for i, d in enumerate(difficulties) if d == level]
        plt.scatter(
            [par_code_lengths[i] for i in idx],
            [par_complexities[i] for i in idx],
            c=color, label=level, alpha=0.6
        )
    plt.title('Code Metrics by Difficulty Level')
    plt.xlabel('Code Length (lines)')
    plt.ylabel('Cyclomatic Complexity')
    plt.legend()
    plt.grid(True)
    plt.savefig('difficulty_scatter_plot.png')
    plt.show()