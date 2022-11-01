#!g1.1
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from melspec import LogMelspec
from metrics import get_au_fa_fr
from metric_tracker import MetricTracker
from random_seed import set_random_seed
from wandb import WanDBWriter

from IPython.display import clear_output


class Trainer:
    def __init__(
        self,
        config,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        use_wandb=True,
        wandb_project="kws_project"
    ):
        set_random_seed(42)

        self.device = config.device

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = optimizer
        self.lr = config.learning_rate

        self.train_log_melspec = LogMelspec(config.sample_rate, config.n_mels, config.device, is_train=True)
        self.val_log_melspec = LogMelspec(config.sample_rate, config.n_mels, config.device, is_train=False)

        self.writer = WanDBWriter(wandb_project, config) if use_wandb else None
        self.train_metric_names = ["loss", "acc"]
        self.val_metric_names = ["loss", "acc", "AU-FA-FR"]

        self.train_history = pd.DataFrame()
        self.val_history = pd.DataFrame()


    def save(self, checkpoint_name):
        os.makedirs(os.path.join(os.getcwd(), 'checkpoints', checkpoint_name), exist_ok=True)
        torch.save(self.model.state_dict(), f'checkpoints/{checkpoint_name}/model.pt')
        torch.save(self.optimizer.state_dict(), f'checkpoints/{checkpoint_name}/optimizer.pt')
        print(f"Saved checkpoint to checkpoints/{checkpoint_name}")
          

    def load(self, checkpoint_name):
        self.model.load_state_dict(torch.load(f'checkpoints/{checkpoint_name}/model.pt'))
        self.optimizer.load_state_dict(torch.load(f'checkpoints/{checkpoint_name}/optimizer.pt'))
        print(f"Loaded checkpoint from checkpoints/{checkpoint_name}")


    def _train_epoch(self, epoch):
        self.model.train()
        len_epoch = len(self.train_dataloader)

        train_metrics = MetricTracker(*self.train_metric_names)
        for batch_num, (x_batch, y_batch) in zip(trange(len_epoch), self.train_dataloader):
            self._process_batch(x_batch, y_batch, self.train_log_melspec, train_metrics, is_train=True)

        train_result = train_metrics.result()

        if self.writer is not None:
            self.writer.add_raw_scalar("epoch", epoch, step=(epoch-1)*len_epoch)
            self.writer.set_step(epoch * len_epoch, mode='train')
            self.writer.add_scalar(
                "learning rate", self.lr
            )
            self._log_metrics(train_result)

        return train_result


    def validate(self):
        self.model.eval()
        all_labels = []
        all_probs = []
        val_metrics = MetricTracker(*self.val_metric_names)
        with torch.no_grad():
            for batch_num, (x_batch, y_batch) in zip(trange(len(self.val_dataloader)), self.val_dataloader):
                probs = self._process_batch(x_batch, y_batch, self.val_log_melspec, val_metrics, is_train=False)
                all_labels.append(y_batch)
                all_probs.append(probs)

        au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
        val_metrics.update("AU-FA-FR", au_fa_fr)

        val_result = val_metrics.result()

        return val_result


    def _validate_epoch(self, epoch):
        val_result = self.validate()
            
        if self.writer is not None:
            len_epoch = len(self.train_dataloader)
            self.writer.set_step(epoch * len_epoch, mode='val')
            self._log_metrics(val_result)

        return val_result


    def _log_metrics(self, metrics):
        for metric_name, metric_val in metrics.items():
            self.writer.add_scalar(f"{metric_name}", metric_val)


    def _process_batch(self, x_batch, y_batch, log_melspec, metrics, is_train):
        batch, labels = x_batch.to(self.device), y_batch.to(self.device)
        batch = log_melspec(batch)

        if is_train:
            self.optimizer.zero_grad()

        # run model # with autocast():
        logits = self.model(batch)
        # we need probabilities so we use softmax & CE separately
        loss = F.cross_entropy(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

        probs = F.softmax(logits, dim=-1)
        argmax_probs = torch.argmax(probs, dim=-1)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

        metrics.update("loss", loss.item())
        metrics.update("acc", acc)
        
        return probs[:, 1].cpu()


    def plot_metric(self, metric='loss'):
        plt.title('{}'.format(metric))

        if metric in self.train_history:
            plt.plot(self.train_history[metric], label='train', zorder=1)
        if metric in self.val_history:
            plt.plot(self.val_history[metric], label='val', zorder=2)
        
        plt.xlabel('train steps')
        
        plt.legend(loc='best')
        plt.grid()

    
    def plot_metrics(self):
        plt.figure(figsize=(15, 4))
        for i, metric in enumerate(set(self.train_metric_names + self.val_metric_names)):
            plt.subplot(1, 3, i + 1)
            self.plot_metric(metric)
        plt.show()


    def train(self, display=True):
        print(f"Start training")
        self.model = self.model.to(self.device)
        for epoch in range(1, self.n_epochs+1):
            print("------------------------------------")
            print('Epoch {}/{}'.format(epoch, self.n_epochs))

            train_result = self._train_epoch(epoch)
            val_result = self._validate_epoch(epoch)
            self.train_history = self.train_history.append(train_result, ignore_index=True)
            self.train_history = self.val_history.append(val_result, ignore_index=True)

            if display:
                clear_output()
                print("------------------------------------")
                print('Epoch {}/{}'.format(epoch, self.n_epochs))
            print()
            epoch_result = train_result
            epoch_result.update({f"{key}_val": value for key, value in val_result.items()})
            print(epoch_result)
            print("------------------------------------")
            print()
            if display:
                self.plot_metrics()
        print("Finish training")
        time.sleep(5)
