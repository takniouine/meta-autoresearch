# autoresearch

## Setup
- Use the existing `TinyStories` dataset.
- Initialize experiments with a base architecture, such as a transformer with a moderate number of layers (e.g., 6-12).

## Experimentation
1. **Establish Baseline:**
   - Start with the base architecture mentioned above.
   - Train the model using the default learning rate (e.g., 3e-5) and a batch size of 32.
   - Focus on training time and validation perplexity (val_bpb) as primary metrics.

2. **Explore Learning Rates:**
   - Run experiments with learning rates in the range of 1e-5 to 5e-5.
   - Use a step size of 10 for incrementing the learning rate.
   - Evaluate the impact of each learning rate on the validation perplexity to identify an optimal learning rate.

3. **Try Different Depths:**
   - Vary the number of layers in the transformer architecture from 6, 8, 10, 12, and 15.
   - Use a fixed number of attention heads (e.g., 8) and hidden size (e.g., 512) for all experiments.
   - Monitor the training time and validation perplexity.

## Output format
```plaintext
Baseline:
- Learning Rate: 3e-5
- Batch Size: 32
- Validation Perplexity: <value>
- Training Time: <value> minutes

Learning Rate Experiments:
- Learning Rate: 1e-5
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Learning Rate: 2e-5
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Learning Rate: 3e-5
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Learning Rate: 4e-5
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Learning Rate: 5e-5
- Validation Perplexity: <value>
- Training Time: <value> minutes

Depth Experiments:
- Layers: 6
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Layers: 8
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Layers: 10
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Layers: 12
- Validation Perplexity: <value>
- Training Time: <value> minutes

- Layers: 15
- Validation Perplexity: <value>
- Training Time: <value> minutes
```

## Logging results
- Log results in a file named `experiment_results.txt` formatted as above.
- Record the datetime of each experiment start and end in the log file.

## The experiment loop
1. Initialize the model with the base architecture.
2. Train and evaluate the model with the default learning rate and batch size.
3. Save the baseline results.
4. For each learning rate in 1e-5 to 5e-5, repeat steps 2-3.
5. For each number of layers in 6 to 15, repeat steps 2-3.
6. Identify the best-performing architecture by analyzing the recorded validation perplexity and training time.
7. Document the findings and the best-performing architecture in the log file.