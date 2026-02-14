import os
import time
import argparse
import numpy as np
import torch

from transformer import *
from bpe import *

"""
运行此脚本前，确保你的 train.bin 和 val.bin 
已经由 Tokenizer 处理好并保存为 uint16 格式
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    # 数据与路径
    parser.add_argument("--data_dir", type=str, required=True, help="包含 train.bin 和 val.bin 的目录")
    parser.add_argument("--out_dir", type=str, default="out", help="模型保存路径")

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1536)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    #训练策略
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=200, help="总迭代次数")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    #评估与日志
    parser.add_argument("--eval_interval", type=int, default=200, help="每隔多少步评估一次验证集")
    parser.add_argument("--eval_iters", type=int, default=100, help="评估时抽样的 batch 数量")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印频率")

    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """
    评估模型在给定数据上的平均 Loss
    """
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        logits = model(X)
        # 展平以计算 cross_entropy: (B*T, V)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def main():
    args = parse_args()
    target_dir = os.path.join(os.getcwd(), "cs336_basics", args.out_dir)
    os.makedirs(target_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 使用 memmap加载数据集 （高效内存利用）
    train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    #val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # 2. 初始化模型
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    # 3. 初始化优化器 
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


    load_checkpoint(os.path.join(target_dir, "best_model.pt"), model, optimizer)

    # 4. 训练循环
    iter_num = 0
    best_val_loss = float('inf')

    print(f"Starting training on {device}...")
    t0 = time.time()

    while iter_num <= args.max_iters:
        # 调整学习率 (Cosine Annealing)
        lr = cosine_lr_schedule(
            iter_num, args.learning_rate, args.min_lr,
            args.warmup_iters, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 获取 Batch 并前向传播
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(X)
        # X, Y : (batch_size, seq_len)
        # logits : (batch_size, seq_len, vocab_size)

        # 计算 Loss (使用交叉熵损失函数 cross_entropy)
        # 展平 logits： (batch_size, seq_len) -> (batch_size * seq_len, vocab_size)
        # 展平 Y: (batch_size, seq_len,) -> (batch_size * seq_len, )
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        if args.grad_clip != 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # 打印日志与评估
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % args.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            val_loss = estimate_loss(model, train_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"STEP {iter_num}: validation loss {val_loss:.4f}")

            # 序列化 Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(target_dir, "best_model.pt")
                save_checkpoint(model, optimizer, iter_num, checkpoint_path)
                print(f"New best model saved to {checkpoint_path}")
        
        iter_num += 1

if __name__ == "__main__":
    main()



