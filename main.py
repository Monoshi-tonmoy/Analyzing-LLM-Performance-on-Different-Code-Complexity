import json
import tree_sitter
from tree_sitter_languages import get_parser
import matplotlib.pyplot as plt
import numpy as np

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
            
            # Count function definitions (starting point)
            if node_type in {'function_definition', 'lambda'}:
                complexity += 1
            
            # Count decision points
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
        return max(1, complexity)  # Minimum complexity is 1
        
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return 0  # Default value for invalid code

def analyze_dataset(dataset_path):
    # Read dataset
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]
    
    code_lengths = []
    complexities = []
    
    for sample in dataset:
        code = sample['code']
        
        # Calculate code length (number of lines)
        lines = code.split('\n')
        code_length = len(lines)
        code_lengths.append(code_length)
        
        # Calculate cyclomatic complexity
        complexity = calculate_cyclomatic_complexity(code)
        complexities.append(complexity)
        
        # Print sample info
        print(f"Sample {sample['id']}:")
        print(f"  Code length: {code_length} lines")
        print(f"  Cyclomatic complexity: {complexity}")
        print()
    
    # Print distributions
    print("\nCode Length Distribution:")
    print(f"  Min: {min(code_lengths)}")
    print(f"  Max: {max(code_lengths)}")
    print(f"  Mean: {np.mean(code_lengths):.2f}")
    print(f"  Median: {np.median(code_lengths)}")
    
    print("\nCyclomatic Complexity Distribution:")
    print(f"  Min: {min(complexities)}")
    print(f"  Max: {max(complexities)}")
    print(f"  Mean: {np.mean(complexities):.2f}")
    print(f"  Median: {np.median(complexities)}")
    
    # Create visualizations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(code_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Code Lengths')
    plt.xlabel('Number of Lines')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(complexities, bins=20, color='salmon', edgecolor='black')
    plt.title('Distribution of Cyclomatic Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('code_metrics_distribution.png')
    plt.show()
    
    # Scatter plot of length vs complexity
    plt.figure(figsize=(8, 6))
    plt.scatter(code_lengths, complexities, alpha=0.6)
    plt.title('Code Length vs Cyclomatic Complexity')
    plt.xlabel('Code Length (lines)')
    plt.ylabel('Cyclomatic Complexity')
    plt.grid(True)
    plt.savefig('length_vs_complexity.png')
    plt.show()

if __name__ == "__main__":
    dataset_path = "cruxeval.jsonl"  # Change to your dataset path
    analyze_dataset(dataset_path)