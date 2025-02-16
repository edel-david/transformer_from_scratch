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
n_genres = 27
n_blocks = 6
n_embd = 126
dropout = 0.1


# context len gets passed by args for some reason

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


class Track:
    def __init__(self, hash, genre, set_name):
        self.hash = hash
        self.genre = genre
        self.set_name = set_name
        self.memmap = None

    def get_path(self, data_dir="data", suffix=".bin"):
        return os.path.join(
            data_dir,
            "tokenized",
            self.set_name,
            f"GENRE_{self.genre}",
            self.hash + suffix,
        )

    def exists(self):
        return os.path.exists(self.get_path())

    def get_memmap(self):
        if self.memmap is None:
            self.memmap = np.memmap(self.get_path(), dtype=np.uint16, mode="r")
        return self.memmap


class Dataset:
    tracks: dict[str, list[Track]]

    def __init__(self, name, data_dir="data", context_length=256):
        self.name = name
        self.sliced_tracks = None

        if not os.path.exists(os.path.join(data_dir, "tokenized", name)):
            raise FileNotFoundError(f"Dataset {name} not found")

        _genres = os.listdir(os.path.join(data_dir, "tokenized", name))
        self.genres = list()

        for genre in _genres:
            if genre.startswith("GENRE_"):
                self.genres.append(genre[6:])

        self.genres = sorted(self.genres)

        assert (
            len(self.genres) == n_genres
        ), f"Expected {n_genres} genres, got {len(self.genres)}"

        self._genres_to_idx = {}
        for i, genre in enumerate(self.genres):
            self._genres_to_idx[genre] = i

        self.tracks = {}

        for genre in self.genres:
            self.tracks[genre] = []
            for file in os.listdir(
                os.path.join(data_dir, "tokenized", name, f"GENRE_{genre}")
            ):
                if file.endswith(".bin"):
                    track = Track(file[:-4], genre, name)
                    if track.exists():
                        self.tracks[genre].append(track)
                    else:
                        raise FileNotFoundError(f"Tokenized file {file} not found")
        
        self.get_slices(context_length=context_length)
        self.genre_probabilities = np.array(
            [len(self.sliced_tracks[genre]) for genre in self.sliced_tracks.keys()]
        )
        self.genre_probabilities = self.genre_probabilities / self.genre_probabilities.sum()



    def genre_to_idx(self, genre):
        return self._genres_to_idx[genre]

    def get_slices(self, context_length):
        if self.sliced_tracks is None:
            self.sliced_tracks = {}
            for genre in self.genres:
                self.sliced_tracks[genre] = []
                for track in self.tracks[genre]:
                    track = track.get_memmap()
                    for i in range(
                        0, len(track) - context_length, context_length // 2
                    ):  # overlap of 50%
                        self.sliced_tracks[genre].append(track[i : i + context_length])
        return self.sliced_tracks

    def get_batch_from_slices(self, batch_size, rng):

        selected_slices = []
        selected_genres = []
        selected_genres_strings = rng.choice(
                list(self.sliced_tracks.keys()), p=self.genre_probabilities
            ,size=(batch_size,))
        for selected_genre in selected_genres_strings:
            selected_genres.append(self.genre_to_idx(selected_genre))
            selected_slice_idx = rng.integers(len(self.sliced_tracks[selected_genre]))
            slice = self.sliced_tracks[selected_genre][selected_slice_idx]
            slice = list(slice)
            slice[0] = 3  # 3 is the genre token
            selected_slices.append(slice)

        x = np.stack(selected_slices)
        y = np.stack(selected_genres)

        return x, y


def compute_gradient(target, prediction, one_hot_lookup):

    target = xp.stack([one_hot_lookup[token] for token in target])

    grad = prediction - target

    grad = grad/np.prod(target.shape[:-1])

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
        n_genres,
        context_length=args.context_length,
        n_layer=n_blocks,
        n_embd=n_embd,
        dropout=dropout,
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

    train_set = Dataset("train", context_length=args.context_length)
    validation_set = Dataset("val", context_length=args.context_length)

    train_set.get_slices(args.context_length)
    validation_set.get_slices(args.context_length)

    def get_batch(set_name):
        if set_name == "train":
            return train_set.get_batch_from_slices(args.batch_size, np_rng)
        if set_name == "val":
            return validation_set.get_batch_from_slices(args.batch_size, np_rng)

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
                # Scale the loss to account for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

                # with open("train_losses.csv", "a") as f:
                #     f.write(f"{iter_num}\t{loss:.8f}\n")
                # disable logging into csv. Use wandb instead

                # Get raw gradient
                grad, target = compute_gradient(
                    Y, logits, one_hot_lookup
                )  # target is Y but one-hot-stacked

                # Continue backward
                

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
                val_runs_with_current_lr+=1
                losses_val = xp.zeros(args.eval_iters)

                top1_correct = 0
                top5_correct = 0
                total_samples = 0

                task_id = progress_step.add_task("Val loss evaluation")

                for k in progress_step.track(
                    range(args.eval_iters), total=args.eval_iters, task_id=task_id
                ):

                    X, Y = get_batch("val")
                    X, Y = cp.asarray(X), cp.asarray(Y)
                    # logits haben Form (Batches, 1, n_genres)?
                    logits, loss = model.forward(X, Y, False)

                    losses_val[k] = loss.item()

                    # descending order
                    sorted_logits = xp.argsort(logits,axis=2)[:,:,::-1]
                    top1_preds = sorted_logits[:,0, 0]
                    top5_preds = sorted_logits[:,0, :5]

                    for i in range(Y.shape[0]):
                        if top1_preds[i].argmax() == Y[i]:
                            top1_correct += 1
                        if Y[i].item() in top5_preds[i]:
                            top5_correct += 1

                    total_samples += Y.shape[0]

                    progress_step.advance(task_id)

                progress_step.remove_task(task_id)

                loss_val_mean = losses_val.mean()
                top1_accuracy = top1_correct / total_samples * 100
                top5_accuracy = top5_correct / total_samples * 100
                all_val_losses.append(loss_val_mean.item())
                wandb.log(
                    {
                        "val_loss": loss_val_mean.item(),
                        #"val_top1_err": 1.0 - top1_accuracy,
                        #"val_top5_err": 1.0 - top5_accuracy,
                        "val_top1_accuracy%": top1_accuracy,
                        "val_top5_accuracy%": top5_accuracy,
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
                else:
                    # check if we should decrease the learning rate
                    if len(all_val_losses) >= 4 and val_runs_with_current_lr > 3:

                        if loss_val_mean.item() > all_val_losses[-2] and loss_val_mean.item() > all_val_losses[-3]:
                            model.set_lr(max( model.lr * 0.5,1e-7))
                            status.update(f"Decreased learning rate to {model.lr}")
                            val_runs_with_current_lr = 0
                            wandb.log({"learning_rate": model.lr}, step=step)
            iter_num += 1

            # termination conditions
            if iter_num > args.epochs:
                break


if __name__ == "__main__":
    main()
