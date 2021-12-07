from time import sleep

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# import torchmetrics
from torch.utils.data import DataLoader

# from pytorch_lightning.loggers import WandbLogger

# from data import CFDataModule


class MyNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(in_features=10, out_features=1)
        # self.metric = torchmetrics.MeanAbsoluteError()
        print("INIT MODEL")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print("got batch of size", x.shape)
        y_hat = self(x).squeeze()
        print(y_hat)
        loss = F.mse_loss(y_hat, y)
        # result = pl.TrainResult(loss)
        # result.log("train_loss", loss, on_epoch=True, sync_dist=True)
        sleep(0.1)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.l1_loss(y_hat, y)
    #     #result = pl.EvalResult(early_stop_on=loss, checkpoint_on=loss)
    #     #result.log("val_loss", loss, sync_dist=True)
    #     return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.0002)
        return torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.0)


class MyData(pl.LightningDataModule):
    def train_dataloader(self):
        cnt_batches = 10
        size_batch = 6
        train_x = [
            torch.rand(
                (size_batch, 10),
                dtype=torch.float32,
                layout=torch.strided,
                device=None,
                requires_grad=False,
            )
            for i in range(cnt_batches)
        ]
        train_y = [
            torch.rand(
                (size_batch,),
                dtype=torch.float32,
                layout=torch.strided,
                device=None,
                requires_grad=False,
            )
            for i in range(cnt_batches)]
        dataset = list(zip(train_x, train_y))
        return DataLoader(dataset, batch_size=None)


def main():
    # wandb_logger = WandbLogger(project="cosmoflow")
    # wandb_logger.log_hyperparams(config)
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    # )
    # print("create tainer")
    print("CREATE TRAINER")
    trainer = pl.Trainer(
        gpus=2,
        strategy="ddp",
        num_sanity_val_steps=0,
        max_epochs=1,
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        # logger=wandb_logger,
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
    )
    # print("tainer created")

    model = MyNet()
    trainer.fit(model, datamodule=MyData())


if __name__ == "__main__":
    main()
