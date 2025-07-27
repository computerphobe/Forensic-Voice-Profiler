import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_CSV = './vocal_features.csv'
OUTPUT_CSV = 'features_labeled.csv'
PLOTS_DIR = 'plots'

def get_label(filename):
    try:
        emotion_code = int(filename.split('-')[2])

        if emotion_code in [4,5,6,7,8]:
            return 1
        elif emotion_code in [1,2,3]:
            return 0
        else:
            return None
    except (IndexError, ValueError):
        return None
    
def explore_data(df):
    print('--Data Exploration--')
    print('\n DataSet shape:', df.shape)
    print(df['label'].value_counts())

def visualize_feat(df):
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    feat_to_plot = ['mean_pitch_hz', 'jitter_local', 'shimmer_local', 'speech_rate_onsets_per_sec']
    print(f"Generating and saving plots to {PLOTS_DIR}/")

    for feat in feat_to_plot:
        plt.figure(figsize=(8,6))
        sns.boxplot(x='label', y=feat, data=df)
        plt.title(f'{feat.replace('-',' '.title())} by label')
        plt.xlabel('Label (0: Neural, 1: stressed)')
        plt.ylabel(feat)

        plot_file = os.path.join(PLOTS_DIR, f'{feat}_boxplot.png')
        plt.savefig(plot_file)
        plt.close()
    print("Completed generating plots for features")

if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("File not Found")
        exit()
    
    df['label'] = df['filename'].apply(get_label)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    explore_data(df=df)
    visualize_feat(df=df)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nLabeled data saved to {OUTPUT_CSV}')
    