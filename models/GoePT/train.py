import sys
import os
import datetime
import argparse
from functools import partial
from collections import deque
from types import NoneType
import json
import time
import cupy as cp
import numpy as np

xp = cp

# from sklearn.metrics import root_mean_squared_error
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from icecream import ic

sys.path.append(".")

from model import GoePT

ic.configureOutput(includeContext=True)
ic.disable()

import wandb


def read_datasets(split, data_dir, context_length, batch_size, rng):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if split == "train":
        data = xp.asarray(
            np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        )
    else:
        data = xp.asarray(
            np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        )

    ix = rng.integers(len(data) - context_length, size=(batch_size,))

    x = np.stack([(data[i : i + context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])

    return x, y


def compute_gradient(target, prediction, one_hot_lookup):
    target = xp.stack([one_hot_lookup[token] for token in target])
    return (prediction - target), target


def get_log_output_table(log_output_buffer: deque) -> Table:

    table = Table()

    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Epoch", style="cyan")
    table.add_column("Train loss", style="green")

    for timestamp, epoch, loss in log_output_buffer:
        table.add_row(f"{timestamp}", f"{epoch}", f"{loss:.5e}")

    return table


def main():

    # Training settings
    parser = argparse.ArgumentParser(description="NanoGPT from scratch")
    parser.add_argument(
        "--data-dir", type=str, default="datasets/tokenized/", help="Dataset directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="models/tokenizers/goe_pt/goe_pt_tokenizer.json",
        help="Vocabulary file",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=32, metavar="N"
    )
    parser.add_argument("--eval-iters", type=int, default=200, metavar="N")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb.init(
        # mode="disabled",  # disable wandb
        # Set the project where this run will be logged
        project="tfs",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"tfs{args.lr}_" + os.uname()[1] + "_" + time.strftime("%Y%m%d-%H%M%S"),
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "transformer",
            "dataset": "goethe",
            "epochs": args.epochs,
        },
    )

    model = GoePT(
        context_length=args.context_length,
        n_layer=6,
        n_embd=384,
        dropout=0.1,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # state_dict = model.state_dict()
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='w', encoding='utf-8') as out_file:
    #     json.dump(state_dict, out_file)
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='r', encoding='utf-8') as in_file:
    #     state_dict = json.load(in_file)
    # model_loaded = GoePT.from_state_dict(state_dict)
    # ic(model_loaded)
    # exit()

    # training loop

    # rng = xp.random.default_rng(args.seed)
    np_rng = np.random.default_rng(args.seed)
    get_batch = partial(
        read_datasets,
        data_dir=args.data_dir,
        context_length=args.context_length,
        batch_size=args.batch_size,
        rng=np_rng,
    )

    # Pre-generate one-hot vectors using the vocab size
    # for gradient computation
    one_hot_lookup = xp.eye(8192)

    iter_num = 0

    best_val_loss = 1e9

    status_console = Console()
    status = status_console.status("[bold green]Starting training...", spinner="runner")
    progress_step = Progress(transient=True)
    header_panel = Panel(Group(status, progress_step))

    log_output_buffer = deque([], maxlen=16)

    table_update_func = partial(
        get_log_output_table, log_output_buffer=log_output_buffer
    )

    step = 0
    # with status_console.screen():
    with Live(header_panel):

        while True:
            # progress_step.console.print(f'Starting epoch: {iter_num + 1}')
            status.update(f"[bold green]Training epoch {iter_num + 1} ...")

            task_id = progress_step.add_task("Training")

            for micro_step in progress_step.track(
                range(args.gradient_accumulation_steps),
                total=args.gradient_accumulation_steps,
                task_id=task_id,
            ):

                X, Y = get_batch("train")
                X, Y = cp.asarray(X), cp.asarray(Y)
                logits, loss = model.forward(X, Y, True)
                wandb.log({"train_loss": loss.item()}, step=step)
                # Scale the loss to account for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

                # with open("train_losses.csv", "a") as f:
                #     f.write(f"{iter_num}\t{loss:.8f}\n")
                # disable logging into csv. Use wandb instead

                # Get raw gradient
                raw_grad, target = compute_gradient(Y, logits, one_hot_lookup)

                # Continue backward
                grad = loss * raw_grad

                model.backward(grad)

                log_output_buffer.append(
                    (
                        datetime.datetime.now().isoformat(),
                        iter_num + 1,
                        loss.item() * args.gradient_accumulation_steps,
                    )
                )

                progress_step.console.clear()
                progress_step.console.print(table_update_func())
                progress_step.advance(task_id)
                step += 1
            progress_step.remove_task(task_id)

            task_id = progress_step.add_task("Updating model")

            model.update()

            progress_step.remove_task(task_id)

            # Evaluate the loss on train/val sets and write checkpoints

            if iter_num % args.eval_interval == 0:

                losses_val = xp.zeros(args.eval_iters)

                task_id = progress_step.add_task("Val loss evaluation")

                for k in progress_step.track(
                    range(args.eval_iters), total=args.eval_iters, task_id=task_id
                ):

                    X, Y = get_batch("val")
                    X, Y = cp.asarray(X), cp.asarray(Y)
                    logits, loss = model.forward(X, Y, False)

                    losses_val[k] = loss.item()

                    progress_step.advance(task_id)

                progress_step.remove_task(task_id)

                loss_val_mean = losses_val.mean()
                wandb.log({"val_loss": loss_val_mean.item()}, step=step)
                if loss_val_mean < best_val_loss:

                    status_update_string = f"Val loss decreased from {best_val_loss:.4f} to {loss_val_mean:.4f}"

                    status_update_string += ". Saving checkpoint..."

                    status.update(status_update_string)

                    checkpoint_path = os.path.join(
                        args.checkpoint_dir, f"goe_pt_iter_{iter_num}.json"
                    )

                    state_dict = model.state_dict()

                    with open(checkpoint_path, mode="w", encoding="utf-8") as out_file:
                        json.dump(state_dict, out_file)

                    status.update(f"Saved checkpoint under {checkpoint_path}")

                    best_val_loss = loss_val_mean

            iter_num += 1

            # termination conditions
            if iter_num > args.epochs:
                break


if __name__ == "__main__":
    main()
