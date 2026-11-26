import math
import torch
import torch.nn as nn


# ==========================================
# PART 1: SCALED DOT-PRODUCT ATTENTION
# ==========================================

def scaled_dot_product_attention(Q, K, V):
    """
    Q = Query matrix
    K = Key matrix
    V = Value matrix

    Returns:
        output = final attention output
        weights = attention probability matrix
    """

    # Find the dimension of each key vector
    d_k = K.size(-1)

    # Step 1: Transpose the Key matrix
    K_transpose = K.transpose(-2, -1)

    # Step 2: Multiply Q with K transpose
    scores = torch.matmul(Q, K_transpose)

    # Step 3: Scale the scores by square root of d_k
    scaled_scores = scores / math.sqrt(d_k)

    # Step 4: Apply Softmax row-wise
    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # Step 5: Multiply attention weights with V
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ==========================================
# PART 2: MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()

        # Model dimension
        self.d_model = d_model

        # Number of attention heads
        self.num_heads = num_heads

        # Every head must have equal dimensions
        assert d_model % num_heads == 0

        # Dimension of each individual head
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key and Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        # Input shape: (batch, sequence_length, d_model)

        batch_size, seq_length, _ = x.shape

        # Step 1: Generate Query, Key and Value
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Split into multiple heads
        # Shape changes:
        # (B, L, d_model) -> (B, L, heads, d_k)

        Q = Q.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        )

        K = K.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        )

        V = V.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        )

        # Step 3: Move the heads dimension
        # (B, L, heads, d_k) -> (B, heads, L, d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Step 4: Calculate attention scores
        # (B, heads, L, d_k) x (B, heads, d_k, L)
        # Result: (B, heads, L, L)

        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        )

        # Step 5: Scale the scores
        scores = scores / math.sqrt(self.d_k)

        # Step 6: Apply Softmax
        attention_weights = torch.softmax(
            scores, dim=-1
        )

        # Step 7: Multiply attention weights with V
        # Result: (B, heads, L, d_k)

        context = torch.matmul(
            attention_weights, V
        )

        # Step 8: Combine all attention heads
        # (B, heads, L, d_k) -> (B, L, heads, d_k)

        context = context.transpose(1, 2).contiguous()

        # Concatenate the heads
        # (B, L, heads, d_k) -> (B, L, d_model)

        context = context.view(
            batch_size, seq_length, self.d_model
        )

        # Step 9: Final linear projection
        output = self.W_o(context)

        return output, attention_weights


# ==========================================
# PART 3: TEST SCALED DOT-PRODUCT ATTENTION
# ==========================================

print("\n========== SCALED DOT-PRODUCT ATTENTION ==========")

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

# Display input
# matrices
print("\nQuery Matrix Q:")
print(Q)

print("\nKey Matrix K:")
print(K)

print("\nValue Matrix V:")
print(V)

# Calculate attention
output, weights = scaled_dot_product_attention(Q, K, V)

print("\nAttention Weights:")
print(weights)

print("\nFinal Scaled Dot-Product Output:")
print(output)


# ==========================================
# PART 4: TEST MULTI-HEAD ATTENTION
# ==========================================

print("\n========== MULTI-HEAD ATTENTION ==========")

# Set seed for reproducible results
torch.manual_seed(42)

# Batch size = 1
# Sequence length = 3
# Model dimension = 4

x = torch.tensor([
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ]
])

# Create a Multi-Head Attention model
d_model = 4
num_heads = 2

mha = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads
)

# Display input
print("\nInput Tensor:")
print(x)

print("\nInput Shape:", x.shape)

# Perform Multi-Head Attention
mha_output, mha_weights = mha(x)

# Display results
print("\nMulti-Head Attention Weights:")
print(mha_weights)

print("\nMulti-Head Attention Output:")
print(mha_output)

print("\nAttention Weights Shape:", mha_weights.shape)
print("Final Output Shape:", mha_output.shape)

# ==========================================
# PART 5: TEST BACKPROPAGATION
# ==========================================

loss = mha_output.sum()

loss.backward()

print("\nGradient Calculation Successful!")

print("\n========== EXECUTION COMPLETED ==========")
