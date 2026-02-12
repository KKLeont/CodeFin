# Requirements
Ensure you have the following installed:

- Python 3.8+
- PyTorch
- Pytorch-Lightning
- matplotlib
- Pandas
- Numpy

# Step 1: Train the Conditional Data Generator (CDG)

Run the following script to train the CDG model:

```bash
python CodeFin/Conditional_Data_Generator/CDG_train.py
```

After training, a checkpoint files for the CDG model will be generated.

# Step 2: Train the Signal-Noise Decomposer (SND)

Run the following script to train the SND model:

```bash
python CodeFin/Signal_Noise_Decomposer/SND_train.py
```

After training, checkpoint files for the SND model will be generated.

# Step 3: Configure checkpoint paths

Edit the file `CodeFin/data_generate.py` to set the paths to the trained CDG and SND checkpoint files. 

# Step 4: Generate data

After configuring the paths, run the following command to generate data:

```bash
python CodeFin/data_generate.py
```

The generated data will be saved to the specified location for further analysis or modeling.

