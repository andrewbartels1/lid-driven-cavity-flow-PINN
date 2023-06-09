# %%
from pathlib import Path
from time import time
from torch.utils.data import RandomSampler, DataLoader
from typing import List, Tuple
import random

# %%
# https://github.com/donny-chan/pinn-torch/blob/5edcd6834a8fddc91db2e9adba958b0b403fd31f/model.py
from typing import List

import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F

import json

# %%
from torch.utils.data import Dataset
from utils import make_text_data_fits_it_sits, dump_json


class PinnDataset(Dataset):
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.examples = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headers = ["x", "y", "t", "p", "u", "v", "Re"]
        return {key: self.examples[idx, i] for i, key in enumerate(headers)}


def get_dataset(data_path: Path) -> Tuple[PinnDataset, PinnDataset]:
    with open(str(data_path)) as data_file:
        data = json.load(data_file)

    # remove the header
    data.pop(0)
    # random.shuffle(data) # turn on random shuffling just uncomment
    print(len(data))

    split_idx = int(len(data) * 0.5)  # train on lower Re, pre
    train_data = data
    test_data = data[split_idx:]

    train_data = PinnDataset(train_data)
    test_data = PinnDataset(test_data)
    return train_data, test_data


def calc_grad(y, x) -> Tensor:
    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


class Pinn(nn.Module):
    """
    `forward`: returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dim. is the predicted values of p, u, v.
    """

    def __init__(self, hidden_dims: List[int]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.ffn_layers = []
        input_dim = 5
        for hidden_dim in hidden_dims:
            self.ffn_layers.append(nn.Linear(input_dim, hidden_dim))
            self.ffn_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.ffn_layers.append(
            nn.Linear(input_dim, 1)
        )  # 2)) # 1 for the Re prediction number
        self.ffn = nn.Sequential(*self.ffn_layers)

        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        p: Tensor = None,
        u: Tensor = None,
        v: Tensor = None,
        Re: Tensor = None,
    ):
        """
        All shapes are (b,)

        inputs: t, u, v
        labels: Re? (p,u,v) is original
        """
        inputs = torch.stack([x, y, t, u, v], dim=1)
        hidden_output = self.ffn(inputs)
        psi = hidden_output[:, 0]
        p_pred = hidden_output[:, 0]
        u_pred = calc_grad(psi, y)
        v_pred = -calc_grad(psi, x)

        preds = torch.stack([p_pred, u_pred, v_pred], dim=1)
        u_t = calc_grad(u_pred, t)
        u_x = calc_grad(u_pred, x)
        u_y = calc_grad(u_pred, y)
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        v_t = calc_grad(v_pred, t)
        v_x = calc_grad(v_pred, x)
        v_y = calc_grad(v_pred, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        p_x = calc_grad(p_pred, x)
        p_y = calc_grad(p_pred, y)

        f_u = (
            u_t
            + self.lambda1 * (u_pred * u_x + v_pred * u_y)
            + p_x
            - self.lambda2 * (u_xx + u_yy)
        )
        f_v = (
            v_t
            + self.lambda1 * (u_pred * v_x + v_pred * v_y)
            + p_y
            - self.lambda2 * (v_xx + v_yy)
        )
        loss = self.loss_fn(u, v, u_pred, v_pred, f_u, f_v)

        return {"preds": preds, "loss": loss, "label": Re}

    # def Re_loss_fn(self, u, v, u_pred, v_pred, f_u_pred, f_v_pred):
    #     """
    #     u: (b, 1)
    #     v: (b, 1)
    #     p: (b, 1)
    #     """
    #     loss = (
    #         F.mse_loss(u, u_pred, reduction="sum")
    #         + F.mse_loss(v, v_pred, reduction="sum")
    #         + F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred), reduction="sum")
    #         + F.mse_loss(f_v_pred, torch.zeros_like(f_v_pred), reduction="sum")
    #     )
    #     return loss

    def loss_fn(self, u, v, u_pred, v_pred, f_u_pred, f_v_pred):
        """
        u: (b, 1)
        v: (b, 1)
        p: (b, 1)
        """
        loss = (
            F.mse_loss(u, u_pred, reduction="sum")
            + F.mse_loss(v, v_pred, reduction="sum")
            + F.mse_loss(f_u_pred, torch.zeros_like(f_u_pred), reduction="sum")
            + F.mse_loss(f_v_pred, torch.zeros_like(f_v_pred), reduction="sum")
        )
        return loss


# %%
torch.random.manual_seed(0)
random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model
hidden_dims = [20] * 8
model = Pinn(hidden_dims=hidden_dims)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
data_path = Path("../data/PINN_input_data.json")
# Data
train_data, test_data = get_dataset(data_path.as_posix())

give_me_my_memory_back = [
    "P_list",
    "U_list",
    "V_list",
    "time_list",
    "dataset",
    "data_ready_to_write",
]
# free up some memory
for var in give_me_my_memory_back:
    globals().pop(var, None)
# next(iter(train_data))

# %% [markdown]
# ### Training Class


# %%
class Trainer:
    """Trainer for convenient training and testing"""

    def __init__(
        self,
        model: Pinn,
        output_dir: Path = None,
        lr: float = 0.0008,
        num_epochs: int = 40,
        batch_size: int = 256,
    ):
        self.model = model

        # Hyperparameters
        self.lr = lr
        self.lr_step = 4  # Unit is epoch
        self.lr_gamma = 0.8
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_interval = 1
        self.samples_per_ep = 1024

        if output_dir is None:
            self.output_dir = Path(
                "result",
                "pinn-large-tanh",
                f"bs{batch_size}"
                f"-lr{lr}"
                f"-lrstep{self.lr_step}"
                f"-lrgamma{self.lr_gamma}"
                f"-epoch{self.num_epochs}",
            )
        else:
            self.output_dir = output_dir

        print(f"Output dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args = {}
        for attr in ["lr", "lr_step", "lr_gamma", "num_epochs", "batch_size"]:
            args[attr] = getattr(self, attr)
        dump_json(self.output_dir / "args.json", args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.lr_gamma
        )

    def get_last_ckpt_dir(self) -> Path:
        ckpt_dirs = list(self.output_dir.glob("ckpt-*"))
        ckpt_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        if len(ckpt_dirs) == 0:
            return None
        return ckpt_dirs[-1]

    def train(self, train_data: PinnDataset, do_resume: bool = True):
        model = self.model
        device = self.device

        # since we are trying to predict a categorical Re, use cross entropy to guide the loss function
        criterion = nn.CrossEntropyLoss()

        sampler = RandomSampler(
            train_data,
            replacement=True,
            num_samples=self.samples_per_ep,
        )
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, sampler=sampler
        )

        print("====== Training ======")
        print(f'device is "{device}"')
        print(f"# epochs: {self.num_epochs}")
        print(f"# examples: {len(train_data)}")
        print(f"# samples used per epoch: {self.samples_per_ep}")
        print(f"batch size: {self.batch_size}")
        print(f"# steps: {len(train_loader)}")
        self.loss_history = []
        model.train()
        model.to(device)

        # Resume
        last_ckpt_dir = self.get_last_ckpt_dir()
        if do_resume and last_ckpt_dir is not None:
            print(f"Resuming from {last_ckpt_dir}")
            self.load_ckpt(last_ckpt_dir)
            ep = int(last_ckpt_dir.name.split("-")[-1]) + 1
        else:
            ep = 0

        train_start_time = time()
        while ep < self.num_epochs:
            print(f"====== Epoch {ep} ======")
            for step, batch in enumerate(train_loader):
                print("step: ", step)

                inputs = {k: t.to(device) for k, t in batch.items()}
                inputs["Re"] = inputs["Re"].type(torch.LongTensor).to(device)

                # Forward
                outputs = model(**inputs)

                # Re categorical prediction
                loss = criterion(outputs["preds"], outputs["label"])
                # loss = outputs["loss"] # original loss computation
                self.loss_history.append(loss.item())

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step % self.log_interval == 0:
                    print(
                        {
                            "step": step,
                            "loss": round(loss.item(), 6),
                            "lr": round(self.optimizer.param_groups[0]["lr"], 4),
                            "lambda1": round(self.model.lambda1.item(), 4),
                            "lambda2": round(self.model.lambda2.item(), 4),
                            "time": round(time() - train_start_time, 1),
                        }
                    )
            self.lr_scheduler.step()
            self.checkpoint(ep)
            ep += 1
        print("====== Training done ======")

    def checkpoint(self, ep: int):
        """
        Dump checkpoint (model, optimizer, lr_scheduler) to "ckpt-{ep}" in
        the `output_dir`,
        and dump `self.loss_history` to "loss_history.json" in the
        `ckpt_dir`, and clear `self.loss_history`.
        """
        # Evaluate and save
        ckpt_dir = self.output_dir / f"ckpt-{ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpointing to {ckpt_dir}")
        torch.save(self.model.state_dict(), ckpt_dir / "ckpt.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
        dump_json(ckpt_dir / "loss_history.json", self.loss_history)
        self.loss_history = []

    def load_ckpt(self, ckpt_dir: Path):
        print(f'Loading checkpoint from "{ckpt_dir}"')
        self.model.load_state_dict(torch.load(ckpt_dir / "ckpt.pt"))
        self.optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))
        self.lr_scheduler.load_state_dict(torch.load(ckpt_dir / "lr_scheduler.pt"))

    def predict(self, test_data: PinnDataset) -> dict:
        batch_size = self.batch_size * 32
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        print("====== Testing ======")
        print(f"# examples: {len(test_data)}")
        print(f"batch size: {batch_size}")
        print(f"# steps: {len(test_loader)}")

        self.model.to(self.device)
        self.model.train()  # We need gradient to predict
        all_preds = []
        all_losses = []
        for step, batch in enumerate(test_loader):
            batch = {key: t.to(self.device) for key, t in batch.items()}
            outputs = self.model(**batch)
            all_losses.append(outputs["loss"].item())
            all_preds.append(outputs["preds"])
        print("====== Testing done ======")
        all_preds = torch.cat(all_preds, 0)
        loss = sum(all_losses) / len(all_losses)
        return {
            "loss": loss,
            "preds": all_preds,
        }


# %% [markdown]
# #### Call the Trainer

# %%
trainer = Trainer(model, batch_size=256, num_epochs=100)
trainer.train(train_data)

# %%
lambda1 = trainer.model.lambda1.item()
lambda2 = trainer.model.lambda2.item()
print(lambda1, lambda2)

# %%
outputs = trainer.predict(test_data)
preds = outputs["preds"]
loss = outputs["loss"]
preds = preds.detach().cpu().numpy()
print("loss:", loss)
print("preds:")
print(preds)

# %%
test_arr = np.array(test_data.data)
p = test_arr[:, 3]
u = test_arr[:, 4]
v = test_arr[:, 5]
p_pred = preds[:, 0]
u_pred = preds[:, 1]
v_pred = preds[:, 2]

# Error
err_u = np.linalg.norm(u - u_pred, 2) / np.linalg.norm(u, 2)
err_v = np.linalg.norm(v - v_pred, 2) / np.linalg.norm(v, 2)
err_p = np.linalg.norm(p - p_pred, 2) / np.linalg.norm(p, 2)

err_lambda1 = np.abs(lambda1 - 1.0)
err_lambda2 = np.abs(lambda2 - 0.01) / 0.01

print(f"Error in velocity: {err_u:.2e}, {err_v:.2e}")
print(f"Error in pressure: {err_p:.2e}")
print(f"Error in lambda 1: {err_lambda1:.2f}")
print(f"Error in lambda 2: {err_lambda2:.2f}")
