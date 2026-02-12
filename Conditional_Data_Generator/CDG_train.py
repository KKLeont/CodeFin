import lightning as pl
from cdg_datalaoder import MyDataloader
import os
from diffusion_model import LightningModel
from pytorch_lightning.loggers import TensorBoardLogger

def run():
    folder_path = './dataset/NDX100'
    file_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
    mydataloader = MyDataloader(file_path, batch_size=800, length=30, num_workers=0)
    dataloader = mydataloader.get_dataloader()
    input_size = mydataloader.get_input_size()
    model = LightningModel(input_size, length = 30, learning_rate=1e-5)
    logger = TensorBoardLogger('tb_logs', name='diffusion')
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=[0,1,2,7], log_every_n_steps=1, logger=logger)
    trainer.fit(model=model, train_dataloaders=dataloader)

if __name__ == "__main__":
    run()
CodeFin/Signal_Noise_Decomposer/SND_train.py