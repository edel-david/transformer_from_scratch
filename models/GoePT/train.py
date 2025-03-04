import sys
import os
import datetime
import argparse
import wandb
from functools import partial
from collections import deque
from types import NoneType
import json
import time
import cupy as cp
import numpy as np

xp = cp
n_genres = 2
n_blocks = 6
n_embd = 204
dropout = 0.1
vocab_size = 483


# context len gets passed by args for some reason

# from sklearn.metrics import root_mean_squared_error
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from icecream import ic

sys.path.append(".")

from layers import Softmax
from model import GoePT
from dataset import Dataset

ic.configureOutput(includeContext=True)
ic.disable()



def compute_gradient(target, prediction, one_hot_lookup):

    target = xp.stack([one_hot_lookup[token] for token in target]).reshape(prediction.shape)

    grad = prediction - target
    # grad = grad/np.prod(target.shape[:-1])
    return grad, target

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
    parser.add_argument("--context-length", type=int, default=512)
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

    only_genres = ["rock", "classical", "pop"]
    train_set = Dataset("train", context_length=args.context_length, uniform=True, only_genres=only_genres)
    validation_set = Dataset("val", context_length=args.context_length, uniform=True, only_genres=only_genres)

    train_set.get_slices(args.context_length)
    validation_set.get_slices(args.context_length)

    n_genres = len(train_set.genres)
    assert len(train_set.genres) == len(validation_set.genres), "Different number of genres in train and validation set"

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
        n_genres,
        vocab_size =vocab_size,
        context_length=args.context_length,
        n_layer=n_blocks,
        n_embd=n_embd,
        dropout=dropout,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # model.transformer["wte"].weight[3].fill(0) # this would deactivate the embedding token embedding
    # model.transformer["wpe"].weight[0].fill(0)

    # state_dict = model.state_dict()
    #with open(os.path.join(args.checkpoint_dir, 'goe_pt_iter_52.json'), mode='w', encoding='utf-8') as out_file:
    #     json.dump(state_dict, out_file)
    #with open(os.path.join(args.checkpoint_dir, 'goe_pt_iter_52.json'), mode='r', encoding='utf-8') as in_file:
    #     state_dict = json.load(in_file)
    #state_dict['n_genres'] = 3
    #model = GoePT.from_state_dict(state_dict)
    # model.set_lr(5e-5)
    # ic(model)
    # exit()

    # training loop

    # rng = xp.random.default_rng(args.seed)
    np_rng = np.random.default_rng(args.seed)

    def get_batch(set_name):
        if set_name == "train":
            return train_set.get_batch_from_slices(args.batch_size, np_rng)
        if set_name == "val":
            return validation_set.get_batch_from_slices(args.batch_size, np_rng)

    def get_batch_triv(set_name):
        x = cp.random.randint(4,7, (args.batch_size,args.context_length))
        selected = cp.zeros((args.batch_size,),dtype=cp.int64)
        selected[:args.batch_size//2]=1
        x = x + selected.reshape((-1,1)) * cp.full((1,args.context_length,),10)
        x[:,0]=3
        return x, selected

    # Pre-generate one-hot vectors using the vocab size
    # for gradient computation
    one_hot_lookup = xp.eye(n_genres)

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
    all_val_losses = []
    val_runs_with_current_lr = 0
    wandb.log({"learning_rate": model.lr}, step=step)
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

                logits, loss = model.forward(X, targets=Y, train=True)
                wandb.log({"train_loss": loss.item()}, step=step)
                # Get raw gradient
                grad, target = compute_gradient(
                    Y, logits, one_hot_lookup
                )  # target is Y but one-hot-stacked
                model.backward(grad)

                log_output_buffer.append(
                    (
                        datetime.datetime.now().isoformat(),
                        iter_num + 1,
                        loss.item()
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
                val_runs_with_current_lr+=1
                losses_val = xp.zeros(args.eval_iters)

                top1_correct = 0
                top5_correct = 0
                total_samples = args.eval_iters * args.batch_size

                task_id = progress_step.add_task("Val loss evaluation")

                for k in progress_step.track(
                    range(args.eval_iters), total=args.eval_iters, task_id=task_id
                ):

                    X, Y = get_batch("val")
                    X, Y = cp.asarray(X), cp.asarray(Y)
                    # logits haben Form (Batches, 1, n_genres)?
                    logits, loss = model.forward(X,targets=Y,train= False)

                    losses_val[k] = loss.item()
                    top1_correct += (cp.argmax(logits,-1).flatten()==Y).sum().item()

                    progress_step.advance(task_id)

                progress_step.remove_task(task_id)

                loss_val_mean = losses_val.mean()
                top1_accuracy = top1_correct / total_samples * 100
                # top5_accuracy = top5_correct / total_samples * 100
                all_val_losses.append(loss_val_mean.item())
                wandb.log(
                    {
                        "val_loss": loss_val_mean.item(),
                        #"val_top1_err": 1.0 - top1_accuracy,
                        #"val_top5_err": 1.0 - top5_accuracy,
                        "val_top1_accuracy%": top1_accuracy,
                        #"val_top5_accuracy%": top5_accuracy,
                    },
                    step=step,
                )

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
                    val_runs_with_current_lr+=1
                else:
                    # check if we should decrease the learning rate
                    if len(all_val_losses) >= 5 and val_runs_with_current_lr > 4:

                        if loss_val_mean.item() > all_val_losses[-2] and loss_val_mean.item() > all_val_losses[-3] and loss_val_mean.item() > all_val_losses[-4]:
                            model.set_lr(max( model.lr * 0.7,1e-5))
                            status.update(f"Decreased learning rate to {model.lr}")
                            val_runs_with_current_lr = 0
                            wandb.log({"learning_rate": model.lr}, step=step)
                        else:
                            val_runs_with_current_lr+=1
                    else:
                        val_runs_with_current_lr+=1
            iter_num += 1

            # termination conditions
            if iter_num > args.epochs:
                break


if __name__ == "__main__":
    main()
