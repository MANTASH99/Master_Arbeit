import matplotlib.pyplot as plt
import numpy as np

# Sample data
frequencies = ['250', '315', '400', '500', '630', '800', '1000', '1250', '1600',
               '2000', '2500', '3150', '4000', '5000', '6300', '8000']

# Define specific important frequencies for each group
group1_important_freq = ['250', '4000']
group2_important_freq = ['2000', '800']
group3_important_freq = ['1000', '8000', '2500']

# Convert frequencies to their corresponding indices in the original list
group1_indices = [frequencies.index(freq) for freq in group1_important_freq]
group2_indices = [frequencies.index(freq) for freq in group2_important_freq]
group3_indices = [frequencies.index(freq) for freq in group3_important_freq]

# Create subplots for each group
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for group 1
axes[0].bar(group1_indices, np.ones(len(group1_indices)), color='red', alpha=0.6)
axes[0].set_title('NAL 50db')
axes[0].set_xlabel('Frequencies (Hz)')
axes[0].set_ylabel('db')
axes[0].set_xticks(range(len(frequencies)))
axes[0].set_xticklabels(frequencies, rotation=45)

# Plot for group 2
axes[1].bar(group2_indices, np.ones(len(group2_indices)), color='blue', alpha=0.6)
axes[1].set_title('NAL 65 db')
axes[1].set_xlabel('Frequencies (Hz)')
axes[1].set_ylabel('db')
axes[1].set_xticks(range(len(frequencies)))
axes[1].set_xticklabels(frequencies, rotation=45)

# Plot for group 3
axes[2].bar(group3_indices, np.ones(len(group3_indices)), color='green', alpha=0.6)
axes[2].set_title('NAL 80db')
axes[2].set_xlabel('Frequencies (Hz)')
axes[2].set_ylabel('db')
axes[2].set_xticks(range(len(frequencies)))
axes[2].set_xticklabels(frequencies, rotation=45)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()