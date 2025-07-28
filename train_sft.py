import argparse
import os
import math
import time
import warnings
from contextlib import nullcontext

import torch 
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel

from transformers import AutoTokenizer

from model.config import YuanMengConfig
from model.model import YuanMengModel
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')

def Logger(content, ddp):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, args, model: YuanMengModel, dataset_loader: DataLoader, ctx, scaler, optimizer, ddp):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    iter_per_epoch = len(dataset_loader) # num_batches
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(dataset_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_time:{}min'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60), ddp)
            
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            ckp = f'{args.out_dir}/sft_{args.hidden_size}.pth'

            if isinstance(model, nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save(state_dict, ckp)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuanMeng Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default='dataset/sft_mini_512.jsonl')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_seed = 1337
    # If you run torchrun --nproc_per_node=8 train_pretrain.py, then the RANK environment variable will be set; 
    # if it equals -1, it means DDP is not enabled.
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")

        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        DEVICE = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(DEVICE)
        args.device = torch.device(DEVICE)
    
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)
    else:
        torch.manual_seed(base_seed)
        torch.cuda.manual_seed(base_seed)

    config = YuanMengConfig(hidden_size=args.hidden_size, num_layers=args.num_layers, max_seq_len=args.max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained('model/yuanmeng_tokenizer')
    model = YuanMengModel(config)
    ckp = f'{args.out_dir}/pretrain_{args.hidden_size}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
    Logger(f'The total number of parameters in the YuanMeng model is: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} million', ddp)

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = DistributedSampler(dataset) if ddp else None
    dataset_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler
    )

    device_type = "cuda" if ddp or "cuda" in args.device else "cpu"
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=device_type)
    scaler = torch.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_epoch(epoch=epoch, args=args, model=model, dataset_loader=dataset_loader, ctx=ctx, scaler=scaler, optimizer=optimizer, ddp=ddp)