1. Dataset
The tutorial uses the Tiny Stories dataset, a popular choice from a 2023 paper that explored how small language models could be while still generating coherent English. Unlike models trained on internet-scale data, Tiny Stories is a specific-task dataset intelligently curated to capture the nuances of English language, grammar, and meaning, suitable for 3-4 year old kids. GPT-4 was used to construct a large amount of these stories.
The primary goal for the 15-million-parameter model is to learn from this dataset to:
• Understand the structure/form of the English language, enabling it to generate grammatically correct sentences.
• Understand meaning, allowing it to create its own stories suitable for young children.
The dataset is publicly available on Hugging Face and is split into training and validation sets. It contains 2 million training stories and about 20,000 validation stories, demonstrating a real-world, not "toy," dataset. An example story provided is: "One day a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp...".
2. Data Pre-processing (Tokenization)
Computers, unlike humans, cannot understand sentences or words directly; they only process numerical data. Therefore, the input text must be converted into a numerical format.
• Inefficiencies of Traditional Tokenization:
    ◦ Character-based tokenization results in a huge number of tokens, exceeding the transformer's attention capacity.
    ◦ Word-based tokenization struggles with spelling mistakes (no mapping to tokens) and creates a very large vocabulary (e.g., 500,000 words in English), which increases computational time and memory for next-token prediction.
• Subword Tokenization (BPE Algorithm): The optimal solution is an intermediate approach called subword tokenization, specifically using the Byte Pair Encoding (BPE) algorithm. BPE maintains a vocabulary of characters, common words, and subwords (parts of words, e.g., "token-ization"). This approach resolves the large vocabulary issue and prevents an excessive number of tokens.
• Tokenizer Function: The tokenizer takes the dataset and converts it into a sequence of tokens, each assigned a unique token ID. The tutorial uses a GPT2 subword tokenizer.
• Efficient Storage of Token IDs: To handle 2 million stories efficiently and prevent RAM overload, all token IDs are merged and stored in a single .bin file (e.g., train.bin, validation.bin) directly on disk using a memory-mapped array. This allows for faster data loading during training and avoids re-tokenization in multiple sessions. The data is processed in batches (e.g., 1024 batches for 2.12 million stories) to speed up the collection of token IDs.
3. Creating Input/Output Pairs
The core purpose of language modeling is next token prediction. Even complex responses from models like ChatGPT are generated one token at a time.
• Context Size (Block Size): This defines the maximum number of tokens the language model looks at simultaneously before predicting the next token. In the example, a context size of four is used, meaning the data is broken into chunks of four tokens. The model actually sees token IDs, not words.
• Batch Size: Data is processed in batches during training to make updates more convenient and prevent excessively long parameter updates.
• Input (X) and Output (Y) Creation: For a given input sequence (e.g., "one day a little"), the corresponding output sequence is simply the input shifted to the right by one token (e.g., "day a little girl"). This means that within one input-output pair, there are multiple next-token prediction tasks. For "one day a little" predicting "day a little girl":
    ◦ If "one" is input, "day" should be output.
    ◦ If "one day" is input, "a" should be output.
    ◦ If "one day a" is input, "little" should be output.
    ◦ If "one day a little" is input, "girl" should be output. The model is trained on all these tasks simultaneously. This seemingly simple objective allows the model to learn the structure and meaning of language.
• Computational Efficiency: Techniques like X.pin_memory() and non_blocking=True are used to lock tensor memory in RAM for faster transfer to the GPU, ensuring the CPU isn't blocked during data copying.
4. Small Language Model Architecture
The architecture is broadly divided into three blocks: Input, Processor (Transformer), and Output.
• Input Block (Token and Positional Embeddings):
    ◦ Token Embedding: Converts each token ID into a higher-dimensional vector (e.g., 768 dimensions in the example, 384 in the code configuration). This vector aims to capture the semantic meaning of the token, allowing words with similar meanings to be closer in the vector space (e.g., "cat" and "dog" are closer than "cat" and "chair"). These embedding values are initially randomized and learned during training. The token embedding matrix serves as a lookup table.
    ◦ Positional Embedding: (Not explicitly detailed in the provided text but implied by adding to token embeddings). This adds information about the position of each token in the sequence.
• Processor (Transformer Block): This is where "magic happens". The input passes through multiple layers (e.g., six layers in the configuration) of these blocks. Each transformer block consists of:
    ◦ Layer Normalization: Applied to stabilize training by ensuring inputs at each layer have a consistent distribution (mean=0, variance=1), preventing "internal covariate shift".
    ◦ Multi-Head Attention (Causal Self-Attention): The core mechanism that enables the model to understand context and relationships between tokens. It allows the model to "pay attention" to relevant tokens when processing another (e.g., understanding if "it" refers to "dog" or "ball").
        ▪ Queries (Q), Keys (K), and Values (V) Matrices: The input embedding vectors are projected into Q, K, and V vectors using trainable weight matrices. These matrices learn how to capture context.
        ▪ Attention Scores: Calculated by taking the dot product of Queries and Keys.
        ▪ Causal Attention: Ensures that a token only pays attention to itself and tokens that come before it in the sequence, preventing the model from "peeking into the future". This is achieved by setting elements above the diagonal of the attention score matrix to negative infinity before applying softmax.
        ▪ Attention Weights: Scores are scaled (by square root of key dimensions for stability) and then passed through a softmax function to convert them into probabilities that sum to one.
        ▪ Context Vectors: Attention weights are multiplied with the Values matrix to produce context vectors. These vectors are "richer" as they now incorporate information from neighboring tokens.
        ▪ Output Projection Layer and Dropout: An optional neural network can be added after attention, followed by a dropout layer to improve generalization.
    ◦ Shortcut/Skip Connections: These are additive connections (input added to output of a block) that provide alternative paths for gradients to flow, preventing the vanishing gradient problem in deep networks.
    ◦ Feed-Forward Neural Network (Multi-Layer Perceptron - MLP): An "expansion-compression neural network" that takes the context vectors, expands them to a higher dimension (e.g., 4 * embedding dimension), and then compresses them back to the original embedding dimension. This allows the model to explore a much richer space and learn new non-linearities, significantly improving performance. It uses the GELU activation function, which is smooth and differentiable. Another dropout layer is applied after this network.
• Output Block (LM Head):
    ◦ Final Layer Normalization: Applied after all transformer blocks.
    ◦ Output Head (LM Head): This is a final neural network that converts the context vectors (size 4x768) into logits (size 4xVocabulary Size, e.g., 4x50257). Each row of the logits matrix represents the model's unnormalized predictions for the next token given the input up to that point.
• Model Configuration: The GPT class is the main model. Its configuration includes:
    ◦ Vocabulary size: 50257
    ◦ Block size (context size): 128
    ◦ Embedding dimensions (n_embedding): 384
    ◦ Dropout rate: 0.1
    ◦ Number of layers (transformer blocks): 6
    ◦ Number of heads (attention heads per block): 6
5. Loss Function
The loss function quantifies how "wrong" the model's predictions are, guiding the learning process.
• Cross-Entropy Loss (Negative Log-Likelihood): This is the main loss function used for language models.
• Calculation:
    1. The model's logits are converted into probabilities using a softmax function. This ensures probabilities for all possible next tokens sum to one.
    2. For each input-output prediction task, the model looks at the probability assigned to the correct target token ID.
    3. The goal is to maximize these probabilities (make them as close to one as possible) for the correct target tokens.
    4. The cross-entropy loss achieves this by taking the negative logarithm of these probabilities. Minimizing this negative log-likelihood loss is equivalent to maximizing the probabilities of the correct tokens.
• Batch Processing: For an entire batch, all logits are flattened, and the cross-entropy loss is calculated across all prediction tasks within that batch.
• The estimate_loss function calculates the mean loss over a prescribed number of evaluation iterations (e.g., 500) for both training and validation data, providing insights into model performance.
6. Training Loop
The training loop orchestrates the forward pass, loss calculation, backpropagation, and parameter updates to minimize the loss.
• Automatic Mixed Precision (torch.autocast): This feature automatically converts numerical precision to float16 (half-precision) for operations where it's safe (e.g., matrix multiplications, dropout) and retains float32 (full-precision) for sensitive operations (e.g., softmax, loss, weight updates). This significantly speeds up computations while maintaining numerical stability.
• Gradient Accumulation: If the desired batch size (e.g., 1024) is too large to fit in GPU memory at once, gradient accumulation is used. The model processes smaller sub-batches (e.g., 32 steps), calculates their gradients, and accumulates these gradients over multiple steps. Parameter updates only occur after the gradients from the specified accumulation steps have been collected, effectively simulating a larger batch size without requiring more GPU memory.
• Optimizer: The AdamW optimizer is used, known for its adaptive learning rate and weight decay, which helps in avoiding local minima and stabilizing training.
• Learning Rate Schedule: A dynamic learning rate is employed, combining a linear warm-up phase (increasing learning rate) followed by a cosine decay (gradually decreasing learning rate). This schedule is often found to offer the "best performance when training language models".
• Training Process Steps:
    1. Get a batch (X, Y): Randomly sample input-output pairs from the dataset.
    2. Forward Pass: Pass the input batch through the model to get the logits.
    3. Calculate Loss: Compute the cross-entropy loss between logits and targets.
    4. Backward Pass (Backpropagation): Compute gradients of the loss with respect to model parameters. These gradients are accumulated if gradient_accumulation_steps is active.
    5. Update Parameters: Only after accumulating gradients for the specified number of steps, the AdamW optimizer updates the model's parameters.
    6. Update Learning Rate: The learning rate is updated according to the schedule.
    7. Evaluate and Print Loss: After a set number of eval_iterations (e.g., 500), the estimate_loss function is called to calculate and print the mean training and validation losses.
    8. Save Best Model: The model's parameters associated with the best validation loss are saved to a file (best_model_params.pt) to prevent loss of progress from computationally intensive training.
• Training Results: After 20,000 iterations (taking ~30-35 minutes on an A100 GPU), both training and validation losses continuously decrease and remain very close, indicating the model is learning well and not overfitting.
7. Inference (Text Generation)
Inference is the process of using the trained model to generate new text.
• Loading the Model: The previously saved best model parameters are loaded to restore the trained model.
• Sequential Generation:
    1. An initial input sentence (e.g., "once upon a time there was a pumpkin") is provided.
    2. This input is passed through the trained model (no parameter updates) to get the logits.
    3. The model looks at the last row of the logits tensor and identifies the token ID with the highest probability as the predicted "next token".
    4. This predicted token is decoded back to text.
    5. The newly generated token is appended to the original input sequence, forming a new, longer input.
    6. This augmented sequence is fed back into the model, and the process repeats until a desired number of new tokens (e.g., 200) are generated. This step-by-step process builds "coherent phrases from the initial input context".
• Inference Strategies:
    ◦ Top-K Sampling: Instead of always picking the single most probable token, the model considers only the k (e.g., 5) most probable tokens and samples from them.
    ◦ Temperature Scaling: A "temperature" factor (applied during softmax) controls the creativity of the output. Higher temperatures increase the entropy of the probability distribution, making the model more likely to choose less probable, more "creative" tokens.
• Generated Stories: Examples show that the model generates grammatically mostly correct sentences. While not perfectly coherent in meaning yet, it's "not random tokens" and tries to form a story. Further training (e.g., 40,000-60,000 iterations) yields much better results.
