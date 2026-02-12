from diffusion_model import LightningModel
from snd_model import LightningModel as SNDModel
import os
import pandas as pd
import numpy as np
import argparse
import torch
from tqdm import tqdm

device = torch.device('cuda:0')
def main(args):
    
    # Load CDG Model checkpoint.
    ckpt_path = 'the ckpt of CDG.'
    model = LightningModel.load_from_checkpoint(ckpt_path, input_size=4, length=30, learning_rate=1e-5, device=device)
    model.to(device)
    model.eval()

    # Load SND Model checkpoint.
    snd_ckpt_path = 'the ckpt of SND.'
    snd_model = SNDModel.load_from_checkpoint(snd_ckpt_path, input_size=4, length=30, learning_rate=1e-5, device=device)
    snd_model.to(device)
    snd_model.eval()

   
    input_folder = './dataset/nasdaq100'  

    # Setting the file save path.
    output_folder = f'./output/data/step{args.inference_steps}_gs{args.guidance_scale}' 
    signal_output_folder = f'./output/signal/step{args.inference_steps}_gs{args.guidance_scale}'
    noise_output_folder = f'./output/noise/step{args.inference_steps}_gs{args.guidance_scale}'

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(signal_output_folder, exist_ok=True)
    os.makedirs(noise_output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_folder, filename)

            ticker = filename.split('.')[0]
            df = pd.read_csv(filepath)
            if df.shape[1] < 6:
                print(f"File {filename} is not formatted correctly, skip the file.")
                continue
            
            ohlc_mean_std = {}
            for col in ['Open', 'High', 'Low', 'Close']:
                ohlc_mean_std[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            
            
            datas=[]
            signals=[]
            noises=[]
            for i in range(10):
                if i < 2:
                    seted_trend = 'Uptrend'
                elif i >= 2 and i < 4:
                    seted_trend = 'Downtrend'
                elif i >= 4 and i < 9:
                    seted_trend = 'Volatile'
                elif i == 9:
                    seted_trend = 'Extreme'
                data = model.sampling_loop(
                    num_inference_steps=args.inference_steps,
                    sequence_length=30,
                    num_features=4,
                    guidance_scale=args.guidance_scale,
                    code=ticker,
                    trend=seted_trend
                )
                datas.append(data.cpu()) 
                # print('data: ', data.shape)
                signal = snd_model.decomposer.signal_encoder(data)
                signals.append(signal.cpu())

                noise = snd_model.decomposer.noise_encoder(data)[0]
                # print('noise: ', len(noise))
                noises.append(noise.cpu())

                reshaped_arrays = [tensor.reshape(30, 4) for tensor in datas]

                combined_array = np.vstack(reshaped_arrays)

                generated_data = pd.DataFrame(combined_array, columns=['Open', 'High', 'Low', 'Close'])


                signal_reshaped_arrays = [tensor.reshape(30, 4).detach().numpy() for tensor in signals]
                signal_combined_array = np.vstack(signal_reshaped_arrays)
                noise_reshaped_arrays = [tensor.reshape(30, 4).detach().numpy() for tensor in noises]
                noise_combined_array = np.vstack(noise_reshaped_arrays)
                generated_signal = pd.DataFrame(signal_combined_array, columns=['Open', 'High', 'Low', 'Close'])
                generated_noise = pd.DataFrame(noise_combined_array, columns=['Open', 'High', 'Low', 'Close'])
            
            for col in ['Open', 'High', 'Low', 'Close']:
                mean = ohlc_mean_std[col]['mean']
                std = ohlc_mean_std[col]['std']
                generated_data[col] = generated_data[col] * std + mean
                generated_signal[col] = generated_signal[col] * std + mean
                generated_noise[col] = generated_noise[col] * std + mean
            
            output_filename = f'diffs_{ticker}.csv'
            output_filepath = os.path.join(output_folder, output_filename)
            sinal_output_filepath = os.path.join(signal_output_folder, output_filename)
            noise_output_filepath = os.path.join(noise_output_folder, output_filename)

            generated_signal.to_csv(sinal_output_filepath, index=False)
            generated_noise.to_csv(noise_output_filepath, index=False)
            generated_data.to_csv(output_filepath, index=False)
            
            print(f'File {filename} dealed as {output_filename}.')
    print('Done.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--guidance_scale',
        default=5,
        type=float
    )
    parser.add_argument(
        '--inference_steps',
        default=50,
        type=int
    )
    args = parser.parse_args()

    main(args)