# 🔵 IMAGE ENCODER
**High Level Summary**
- This is a CNN based architecture . 
- Uses ReLU for Activation
- **Standard CNN Input Shape:** is : [B,C, H, W] = [batch_size, num_channels, height, width]

**Details**
- CNN based architecture 
- **Standard CNN Input Shape:** is : [B,C, H, W] = [batch_size, num_channels, height, width]
- **Layer Norm:**  when this has to be used, you'd have to flatten the input appropriately. Also Layer Norm is typically used in the context of Transformers, not sure why it is being used in the Image Encoder
- **Flatten:**  know which dimension to start at. [B,C,H,W] = [B, C,1,1]  has to become [B,C]. When dim =1 , means start flattening from dimension =1 (leave dimension 0 = Batch size unaltered). so [B,C,1,1] = [B,C*1*1] = [B,C]
- **ReLU Activation:** CNNs use ReLUs typically (not a hard rule). 
  - ReLU(x)=max(0,x)
  - ReLU is hard gating: “Either pass the signal or kill it.” 
  - ReLU acts like: “This edge / corner / texture is present or not.” That’s exactly what convolutional filters want.
  - Very cheap to compute. Strong sparsity (many zeros). Sharp nonlinearity. Stable gradients
- **ReLU inplace:** operation is supported

<br><br><br>

# 🔵 TEXT ENCODER
**High Level Summary**
- Transformer based architecture .
- Mutli Head Attention has no activations. Softmax is the Non Linearity
- Multi Layer Perceptron has GELU Activations
- **Standard Transformer Input Shape:** is : [B,S,E] = [batch_size, sequence_length, embedding_dimension]
- So all modules, Multi Layer Perceptrons, Multi Attention Heads all use the same shape of X
  
**Details**
## 🟢 SHA: SINGLE HEAD ATTENTION
**i) sha_embed_dim:** 
- Usually referred to as hidden dimension . sha_embed_dim != mha_embed_dim. Its rather a factor of it.
- The mha_embed_dim gets split into multiple single attention heads.
- As a result the input x would also have to split into the attention heads. But instead of splitting it , you'd just use a linear layer to project from mha_embed_dim to sha_embed_dim. 
      
**ii) Tensor Transpose:**  
 This is not a mere transpose of a AxB matrix. This is a tensor. Hence you would have to mention which dimensions to transpose. K.transpose(-2,-1) = K.transpose(1,2).

**iii) Softmax:**  
- know which dimension to apply softmax.
- Remember the Query vs Key matrix in Dr. Sreedat Panat's lecture 3.2. This is a square matrix of size [B,S,S] .Each row belongs to one Query. Each column is one Key.  
- Query: dimension = [B,S,E]
- Key: dimension = [B,S,E], Key Transpose Dimension = [B,E,S]
- Q@K.Transpose(-2,-1) dimension : [B,S,S]
- The first S belongs to Query, the Last S belongs to K. Take the softmax across the Keys. Softmax is taken for every horizontal query(row) , across the keys(columns). So dimension is the column 

**iv) No Activations inside Attention Heads:**
- Softmax is the only non linearity needed inside attention heads.
- Any additional non linearity will come later in the MLP layers (in the form of GELU)

## 🟢 MHA: MUTI HEAD ATTENTION
**i) mha_embed_dim(embed_dim) vs sha_embed_dim:** 
- mha_embed_dim is more commonly called embedding dimension. I call it mha_embed_dim to differentiate from sha_embed_dim.
- The mha_embed_dim gets split into multiple single attention heads. Example mha_embed_dim(embed_dim) = 64 . num_attention_heads = 4. The sha_embed_dim would be 16.
- As a result the input x would also have to split into the attention heads. But instead of splitting it , you'd just use a linear layer to project from mha_embed_dim to sha_embed_dim. 
- Example. Input X would also need to get split into 4 parts. Example X = [B, S, E] = [1000, 5, 64] . Where B = Batch Size = 1000. S = Sequence Length , Number of Tokens, Number of Words = 5. "You must dream big always" . So it needs to become [1000,5,16], [1000,5,16],[1000,5,16],[1000,5,16]. But instead of getting split like this. Q, K,V linear layers just map it from 64 to 16 inside the single head attention. 
    
**ii) out_proj layer . Why is it needed:**  Dimension 
- This is a "per token" feature transformer. Hence non linearities such as GELU is needed \
  
**ii) GELU Activation:**
  - Transformers use GELUs typically (not a hard rule).
  - GELU(x)=x⋅Φ(x) , where Φ(x) is the CDF of a standard Gaussian.
  - GELU is Soft gating (Probabilistic Gating): “Pass the signal proportionally based on how confident it is.”
  - Smooth,Nonzero gradient everywhere,No hard cutoff at 0 .
  - Transformers use Smooth continuous operations: QKᵀ → softmax → weighted sums. GELU is smooth and continuous
  - Transformers have a lot of Residual Operations: x ← x + Attention(x) &x ← x + MLP(x).With these residuals, if ReLU is used there are a lot zeros and ReLU updates can become too binary.  - GELU allows small negative updates in the residuals as well. This leads to better stability . It preserves information better.
- **GELU inplace:** operation is **NOT SUPPORTED**. GELU cannot be done inplace


