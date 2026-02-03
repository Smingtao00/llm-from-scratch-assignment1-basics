import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float,Bool,Int
import math
import typing
import os

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x is row vector
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int , embedding_dim: int, device=None, dtype=None):
        """
        Args:
            num_embeddings: size of the vocabulary 词表大小
            embedding_dim: Dimension of the embedding vectors, i.e., d_model 词嵌入向量维度 
            device: torch.device, Device to store the parameters on
            dtype: torch.dtype, Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal_(self.weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup for the given token IDs
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value of the model
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        # and return a tensor of the same shape
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        result = x_norm * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #W1x = torch.einsum("ij,...j->...i", self.w1.weight, x)
        W1x = self.w1(x)
        sig_W1x = torch.sigmoid(W1x)
        SiLU_W1x = torch.einsum("...i,...i->...i", W1x, sig_W1x)
        #W3x = torch.einsum('ij,...j->...i', self.w3.weight, x)
        W3x = self.w3(x)
        SiLU_W1x_times_W3x = torch.einsum('...i,...i->...i', SiLU_W1x, W3x)
        #SwiGLU_x = torch.einsum('ij,...j->...i', self.w2.weight, SiLU_W1x_times_W3x)
        SwiGLU_x = self.w2(SiLU_W1x_times_W3x)
        return SwiGLU_x


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k, max_seq_len:int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        position = torch.arange(max_seq_len, device=device).float()
        idx = torch.arange(0, d_k, 2, device=device).float() / d_k
        inv_freq = 1.0 / (theta ** idx)

        angles = torch.einsum("i,j->ij", position, inv_freq)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # (max_seq_len, d_k/2)

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions] 
        sin = self.sin_cached[token_positions]
        # (..., seq_len, d_k/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # (..., seq_len, d_k/2)

        x_rotated_even = torch.einsum('...i,...i->...i', x_even, cos) - torch.einsum('...i,...i->...i', x_odd, sin)
        x_rotated_odd = torch.einsum('...i,...i->...i', x_even, sin) + torch.einsum('...i,...i->...i', x_odd, cos)

        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals = torch.max(x, dim = dim, keepdim=True).values
    x_stabel = x - max_vals
    exp_x = torch.exp(x_stabel) 
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    softmax = exp_x / sum_exp

    return softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask=None
) -> torch.Tensor:
    """
    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., values, d_v)
        mask: (..., queries, keys) 
    Output:
        (..., querys, d_v)
    """
    d_k = Q.shape[-1]
    attention_scores = torch.einsum('...qd,...kd->...qk', Q, K)
    attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == False, float('-inf'))
    
    attention_weights = softmax(attention_scores, dim=-1)
    output = torch.einsum('...qk,...kv->...qv', attention_weights, V)

    return output


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        use_rope: bool =False,
        device=None,
        dtype=None
    ):
        super().__init__();
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        self.d_v = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)

        if use_rope:
            self.rope = RoPE(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        x: (..., sequence_len, d_in(d_model))
        positions: (..., sequence_len)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]

        Q = self.q_proj(x) 
        K = self.k_proj(x)
        V = self.v_proj(x)
        # (..., seq_len, num_heads * d_k/d_v)

        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v)

        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        if self.use_rope and self.rope is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
                for _ in range(len(batch_shape)):
                    positions = positions.unsqueeze(0)
                positions = positions.expand(*batch_shape, seq_len)
            
            Q_shape = Q.shape
            K_shape = K.shape 

            Q_for_rope = Q.reshape(-1, seq_len, self.d_k)
            K_for_rope = K.reshape(-1, seq_len, self.d_k)

            pos_for_rope = positions.unsqueeze(-2).expand(*batch_shape, self.num_heads, seq_len)
            pos_for_rope = pos_for_rope.reshape(-1, seq_len)

            Q_for_rope = self.rope(Q_for_rope, pos_for_rope)
            K_for_rope = self.rope(K_for_rope, pos_for_rope)

            Q = Q_for_rope.reshape(Q_shape)
            K = K_for_rope.reshape(K_shape)
        
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device),
            diagonal=0
        )
        mask = mask.view(1, 1, seq_len, seq_len)
        target_shape = (*batch_shape, self.num_heads, seq_len, seq_len)
        mask = mask.expand(target_shape)

        attention_output = scaled_dot_product_attention(Q, K, V, mask)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(*batch_shape, seq_len, self.num_heads * self.d_v)

        output = self.output_proj(attention_output)

        return output


class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = multihead_self_attention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff
        )
    
    def forward(
        self,
        x: torch.Tensor
    ):
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))
        

class transformer_lm(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.layers = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(
            d_model=d_model
        )

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        x: Int(Tensor, ["batch_size sequence_length"])
        """
        y = self.token_embeddings(x)
        # (batch_size, seq_len, d_model)

        for layer in self.layers:
            y = layer(y)

        y = self.ln_final(y)

        logits = self.lm_head(y)
        # (batch_size, seq_len, vocab_size)
        return logits


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: (batch_size, vocab_size)
    target: (batch_size, )
    """
    batch_size, vocab_size = logits.shape

    max_logits = torch.max(logits, dim=1, keepdim=True).values
    logits_stable = logits - max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=1))
    target_logits = logits_stable[torch.arange(batch_size), targets]

    losses = -target_logits + log_sum_exp

    return torch.mean(losses)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter : ({betas[0], betas[1]})")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay valud: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                lr_t = lr * math.sqrt(bias_correction2) / bias_correction1 

                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.add_(exp_avg/denom, alpha=-lr_t)
                p.data.add_(p.data, alpha=-lr*weight_decay)

        return loss


def cosine_lr_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * 0.5 * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    eps = 1e-6
    gradients = []
    for param in parameters:
        if param.grad is not None:
            gradients.append(param.grad)
    
    if not gradients:
        return
    
    total_norm = 0.0
    for grad in gradients:
        total_norm += grad.norm().item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in gradients:
            grad.mul_(clip_coef)


def get_batch(dataset, batch_size: int, context_length: int, device: str):
    import numpy as np

    max_start_idx = len(dataset) - context_length

    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    input_sequences = []
    target_sequences = []

    for start_idx in start_indices:
        input_seq = dataset[start_idx:start_idx + context_length]
        target_seq = dataset[start_idx + 1:start_idx + context_length + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    input_array = np.array(input_sequences)
    target_array = np.array(target_sequences)

    input_tensor = torch.from_numpy(input_array).long().to(device)
    target_tensor = torch.from_numpy(target_array).long().to(device)

    return input_tensor, target_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']