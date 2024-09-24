
### Positional Encoding

This repository contains the implementation of a 'Positional Encoding' class in PyTorch. Positional Encoding is commonly used in transformer models to inject sequence information into the model, as transformers lack inherent knowledge of the sequence order of the input.

### Description

The 'Positional Encoding' class provides a way to encode the positional information of tokens in a sequence using sinusoidal functions. The encoding alternates between sine and cosine functions of different frequencies. This method was first introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

### Initialization Parameters

- `embed_size` (*int*): The size of embedding (dimensionality) for each position in the input sequence.
- `max_len` (*int*, default = 512): The maximum length of the input sequence for which positional encoding is computed.

### Methods

#### `__init__(self, embed_size, max_len=512)`

The constructor initializes the positional encoding matrix. It computes the encoding using sinusoidal functions for all possible positions up to `max_len` and for all embedding dimensions up to `embed_size`.

#### `forward(self, x)`

Adds the positional encoding to the input tensor `x`. The encoding ensures that the input maintains positional information.

- **Parameters**: 
  - `x` (*Tensor*): A tensor of shape `[batch_size, sequence_length, embed_size]`.
  
- **Returns**: 
  - The input tensor `x` with the positional encoding added to it.

---

### Formulas for Positional Encoding

The positional encoding is calculated using sine and cosine functions based on the position of the token in the sequence and the embedding dimension.

For even indices (using sine function):
```plaintext
PE(pos, 2i) = sin(pos / 10000^(2i / embed_size))
```

For odd indices (using cosine function):
```plaintext
PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_size))
```

Here:
- `pos` is the position of the token in the sequence (ranging from 0 to `max_len - 1`).
- `i` is the index of the embedding dimension.
- `embed_size` is the total dimensionality of the embedding space.
- The `10000` term acts as a scaling factor to control the frequency of the sine and cosine waves.


