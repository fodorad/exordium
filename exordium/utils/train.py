import os
import pprint
import pickle
from pathlib import Path
from argparse import ArgumentParser
import psutil
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def get_parameters(model: torch.nn.Module):
    return filter(lambda p: p.requires_grad, model.parameters())


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TrainWrapper(pl.LightningModule):

    def __init__(self, hparams: dict, model, criterion, metrics):
        super(TrainWrapper, self).__init__()

        self.hparams = hparams
        self.model = model
        self.criterion = criterion
        self.metrics = metrics

        self.model_checkpoint = {
            'loss': float('inf')
        }
        self.history = {'train': [], 'valid': [], 'test': []}
        self.save_hyperparameters(hparams, ignore=["class_weights"])


    def load_model(self, weights_path: str, device: str = 'cuda:0'):
        self.model.to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)


    def forward(self, left_patch, right_patch):
        return self.model(left_patch, right_patch)


    def shared_step(self, batch, batch_idx):
        left, right, y_true = batch
        y_logits = self.forward(left, right)
        y_pred = torch.sigmoid(y_logits)
        loss = self.criterion(y_logits, y_true)
        return loss, y_true, y_pred


    def training_step(self, batch, batch_idx):
        loss, y_true, y_pred = self.shared_step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, on_epoch=True)
        return {'loss': loss, 'y_true': y_true, 'y_pred': y_pred}


    def validation_step(self, batch, batch_idx):
        loss, y_true, y_pred = self.shared_step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, on_epoch=True)
        return {'loss': loss, 'y_true': y_true, 'y_pred': y_pred}


    def test_step(self, batch, batch_idx):
        loss, y_true, y_pred = self.shared_step(batch, batch_idx)
        self.log_dict({"test_loss": loss}, on_epoch=True)
        return {'loss': loss, 'y_true': y_true, 'y_pred': y_pred}


    def save_model(self, metrics, output_dir):
        Path(self.hparams.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if metrics is None: return
        if metrics['loss'] <= self.model_checkpoint['loss']:
            path = Path(output_dir) / "model_best_loss.pt"
            print(f'loss decreased from {self.model_checkpoint["loss"]} to {metrics["loss"]}\nModel saved to {path}')
            self.model_checkpoint['loss'] = metrics['loss']
            torch.save(self.model.state_dict(), path)


    def calculate_metrics(self, step_outputs, phase: str):
        y_pred = torch.cat([d['y_pred'] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy()
        y_true = torch.cat([d['y_true'] for d in step_outputs], dim=0).detach().cpu().squeeze().numpy()
        mean_loss = torch.stack([d['loss'] for d in step_outputs]).mean().detach().cpu().numpy()
        metrics = self.metrics(y_true, y_pred)
        if metrics is not None:
            metrics['loss'] = mean_loss
            self.history[phase].append(metrics)
            print(f'\n\n[{phase}]:')
            pprint.pprint(metrics)
        return metrics


    def training_epoch_end(self, validation_step_outputs):
        self.calculate_metrics(validation_step_outputs, 'train')


    def validation_epoch_end(self, training_step_outputs):
        val_metrics = self.calculate_metrics(training_step_outputs, 'valid')
        self.save_model(val_metrics, self.hparams.checkpoint_dir)


    def test_epoch_end(self, test_step_outputs):
        self.calculate_metrics(test_step_outputs, 'test')


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5, verbose=True),
                'monitor': 'val_loss',
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir: str | Path):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', default=1e-3, type=float)
        parser.add_argument('--checkpoint_dir', default=str(root_dir / 'models'), type=str)
        parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
        parser.add_argument('--max_epochs', type=int, default=100, help="Maximum number of epochs to perform; the trainer will Exit after.")
        return parser


if __name__ == "__main__":
    root_dir = Path('.').resolve()

    root_parser = ArgumentParser()
    root_parser.add_argument('--gpu', type=int, default=2, help='gpu to use')
    root_parser.add_argument('--db', type=str, choices=["rt-bene"], default="rt-bene")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--seed', type=int, default=42)
    root_parser.add_argument('--test', action="store_true", help="Test only")
    root_parser.add_argument('--save_dir', type=str, default=str(root_dir / 'logs'))
    model_parser = TrainWrapper.add_model_specific_args(root_parser, root_dir)
    hparams = model_parser.parse_args()

    seed_everything(hparams.seed)
    print('Hyperparams:', hparams)

    datamodule = ...
    model = ...
    criterion = ...
    metrics = ...

    save_dir = Path(hparams.save_dir) / 'logs'
    save_dir.mkdir(parents=True, exist_ok=True)
    hparams.checkpoint_dir = save_dir / 'models'

    if not hparams.test:
        print('Class weights:', datamodule.class_weights)
        model_wrapper = TrainWrapper(hparams=hparams, model=model, criterion=criterion, metrics=metrics)

        cp_val_loss_callback = ModelCheckpoint(monitor='val_loss', dirpath=str(save_dir), filename='best_val_loss', mode='min')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=True, mode='min')
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(save_dir))
        csv_logger = pl_loggers.CSVLogger(save_dir, name='metrics.csv')

        trainer = Trainer(default_root_dir=save_dir,
                          accelerator="gpu",
                          devices=[hparams.gpu],
                          callbacks=[cp_val_loss_callback, early_stop_callback],
                          min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          logger=[tb_logger, csv_logger],
                          benchmark=True)

        trainer.fit(model_wrapper, datamodule)
            
        history_path = save_dir / 'history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(model_wrapper.history, f)
            print(f'History saved to {str(history_path)}')
        
        del trainer, model_wrapper
    print('Training is finished...')
