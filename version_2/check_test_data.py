import pandas as pd
import json
import os

# Check full Phase 2 data
test_data_path = 'results/phase1_outputs/phase1_24h_windows_CLEAN.csv'
df = pd.read_csv(test_data_path)
print('Full Phase 2 data:')
print(f'  Total samples: {len(df)}')
print(f'  Total deaths: {int(df["mortality"].sum())}')
print(f'  Mortality rate: {df["mortality"].mean()*100:.2f}%')

# Check if there's a diagnostics file with test split info
try:
    with open('results/phase2_outputs/diagnostics_CORRECTED.json', 'r') as f:
        diag = json.load(f)
        print('\nPhase 2 model diagnostics:')
        for key in ['test_auc', 'test_samples', 'test_deaths', 'mortality_rate']:
            print(f'  {key}: {diag.get(key, "N/A")}')
except Exception as e:
    print(f'\nNo diagnostics: {e}')

# Check for saved test data files
print('\nSearching for test data files...')
for filename in os.listdir('results/phase2_outputs'):
    if 'test' in filename.lower() and filename.endswith('.csv'):
        path = f'results/phase2_outputs/{filename}'
        test_df = pd.read_csv(path)
        print(f'\n  {filename}:')
        print(f'    Samples: {len(test_df)}')
        if 'mortality' in test_df.columns:
            print(f'    Deaths: {int(test_df["mortality"].sum())}')
            print(f'    Mortality rate: {test_df["mortality"].mean()*100:.2f}%')
