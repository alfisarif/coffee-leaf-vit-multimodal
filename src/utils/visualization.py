import matplotlib.pyplot as plt

def plot_training_history(history_csv, output_path, title):
    import pandas as pd
    df=pd.read_csv(history_csv)
    plt.figure(figsize=(7,4.5)); plt.plot(df['epoch'],df['train_accuracy'],label='Train accuracy'); plt.plot(df['epoch'],df['val_accuracy'],label='Validation accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(output_path,dpi=300); plt.close()
