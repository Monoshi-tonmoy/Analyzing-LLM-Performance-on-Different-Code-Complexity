import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load JSON data
with open('/home/monoshi/Analyzing-LLM-Performance-on-Different-Code-Complexity/LLM Utils/Results_input_output/input_output_predictions.json', 'r') as f:
    data = json.load(f)

# Extract model names and metrics
models = list(data.keys())
metrics = {
    "input": {
        "overall": [],
        "easy": [],
        "medium": [],
        "hard": []
    },
    "output": {
        "overall": [],
        "easy": [],
        "medium": [],
        "hard": []
    }
}

for model in models:
    metrics["input"]["overall"].append(data[model]["pt0"]["python"]["input"]["overall_accuracy"])
    metrics["output"]["overall"].append(data[model]["pt0"]["python"]["output"]["overall_accuracy"])
    for difficulty in ["easy", "medium", "hard"]:
        metrics["input"][difficulty].append(
            data[model]["pt0"]["python"]["input"]["detailed_result"]["by_difficulty"][difficulty]["accuracy"]
        )
        metrics["output"][difficulty].append(
            data[model]["pt0"]["python"]["output"]["detailed_result"]["by_difficulty"][difficulty]["accuracy"]
        )

# Shorten model names for plotting
short_names = [name for name in models]

# Set seaborn style (this is the correct way to use seaborn styling)
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Common style parameters
bar_width = 0.25
fontsize = 12
rotation = 45
ha = 'right'

# Plot 1: Overall Input vs. Output Accuracy
fig1, ax1 = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35
rects1 = ax1.bar(x - width/2, metrics["input"]["overall"], width, 
                label='Input Prediction', color='#1f77b4', edgecolor='black')
rects2 = ax1.bar(x + width/2, metrics["output"]["overall"], width, 
                label='Output Prediction', color='#ff7f0e', edgecolor='black')

ax1.set_ylabel('Accuracy', fontsize=fontsize)
ax1.set_title('Overall Input vs. Output Prediction Accuracy', fontsize=fontsize+2, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=rotation, ha=ha, fontsize=fontsize)
ax1.legend(fontsize=fontsize)

# Add value labels
for rect in rects1 + rects2:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=fontsize-2)

fig1.tight_layout()
fig1.savefig('overall_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Output Accuracy by Difficulty (Grouped Bars)
fig2, ax2 = plt.subplots(figsize=(14, 7))
x = np.arange(len(models))
width = 0.25

rects_easy = ax2.bar(x - width, metrics["output"]["easy"], width, 
                    label='Easy', color='#2ca02c', edgecolor='black')
rects_medium = ax2.bar(x, metrics["output"]["medium"], width, 
                      label='Medium', color='#ff7f0e', edgecolor='black')
rects_hard = ax2.bar(x + width, metrics["output"]["hard"], width, 
                    label='Hard', color='#d62728', edgecolor='black')

ax2.set_ylabel('Accuracy', fontsize=fontsize)
ax2.set_title('Output Prediction Accuracy by Difficulty Level', fontsize=fontsize+2, pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, rotation=rotation, ha=ha, fontsize=fontsize)
ax2.legend(fontsize=fontsize)

# Add value labels
for rects in [rects_easy, rects_medium, rects_hard]:
    for rect in rects:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=fontsize-2)

fig2.tight_layout()
fig2.savefig('output_by_difficulty_grouped.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Input Accuracy by Difficulty
fig3, ax3 = plt.subplots(figsize=(14, 7))
x = np.arange(len(models))
width = 0.25

rects_easy = ax3.bar(x - width, metrics["input"]["easy"], width, 
                    label='Easy', color='#2ca02c', edgecolor='black')
rects_medium = ax3.bar(x, metrics["input"]["medium"], width, 
                      label='Medium', color='#ff7f0e', edgecolor='black')
rects_hard = ax3.bar(x + width, metrics["input"]["hard"], width, 
                    label='Hard', color='#d62728', edgecolor='black')

ax3.set_ylabel('Accuracy', fontsize=fontsize)
ax3.set_title('Input Prediction Accuracy by Difficulty Level', fontsize=fontsize+2, pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, rotation=rotation, ha=ha, fontsize=fontsize)
ax3.legend(fontsize=fontsize)

# Add value labels
for rects in [rects_easy, rects_medium, rects_hard]:
    for rect in rects:
        height = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=fontsize-2)

fig3.tight_layout()
fig3.savefig('input_by_difficulty.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Heatmap of All Accuracies
fig4, ax4 = plt.subplots(figsize=(14, 8))
heatmap_data = np.array([
    metrics["input"]["easy"] + metrics["output"]["easy"],
    metrics["input"]["medium"] + metrics["output"]["medium"],
    metrics["input"]["hard"] + metrics["output"]["hard"]
]).T

sns.heatmap(heatmap_data, annot=True, fmt=".3f", 
            xticklabels=['In-Easy', 'In-Medium', 'In-Hard', 'Out-Easy', 'Out-Medium', 'Out-Hard'],
            yticklabels=short_names, cmap="YlGnBu", cbar_kws={'label': 'Accuracy'},
            ax=ax4, annot_kws={"size": fontsize})

ax4.set_title('Input/Output Accuracy by Model and Difficulty', fontsize=fontsize+2, pad=20)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=rotation, ha=ha, fontsize=fontsize)
ax4.set_yticklabels(ax4.get_yticklabels(), fontsize=fontsize)
ax4.figure.axes[-1].yaxis.label.set_size(fontsize)  # Colorbar label size

fig4.tight_layout()
fig4.savefig('accuracy_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()