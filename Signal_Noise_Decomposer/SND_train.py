import lightning as pl
from snd_dataloader import MyDataloader
import os
from snd_model import LightningModel
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def run():
    folder_path = './dataset/NDX100'
    file_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    mydataloader = MyDataloader(file_path, batch_size=1024, length=30, num_workers=0)
    dataloader = mydataloader.get_dataloader()
    input_size = mydataloader.get_input_size()
    model = LightningModel(input_size, length = 30, learning_rate=1e-5)
    logger = TensorBoardLogger('tb_logs', name='snd_model')
    checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='{epoch}-{step}',
    save_top_k=-1,
    every_n_train_steps=100
)
    trainer = pl.Trainer(max_epochs=2, accelerator="gpu", devices=[0], log_every_n_steps=1, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=dataloader)

if __name__ == "__main__":
    run()
