import pandas as pd
import re

position = "boundary"
# Load Data
radiomics_df = pd.read_csv(f"Only_radiomics_based_classification/radiomics_features_washu2_p1_143_{position}.csv")
gt_df = pd.read_excel(r"data\PAT_imaging_record.xlsx", sheet_name="ROI STATS V4 (3)")

#print(radiomics_df['ImagePath'].str.extract(r"/p(\d+)/([A-Z])/", expand=True).value_counts())

# Extract patient ID and side
def extract_pid_side(path):
    match = re.search(r"/p(\d+)/([LRC])/", str(path))
    if match:
        pid = int(match.group(1))
        side = match.group(2)
        #print(side)
        return pd.Series({'PatientID': pid, 'Side': side})
    else:
        return pd.Series({'PatientID': None, 'Side': None})

radiomics_df[['PatientID', 'Side']] = radiomics_df['ImagePath'].apply(extract_pid_side)

# Prepare GT data
gt_df = gt_df[['GT', 'Patient ID', 'Side']].dropna(subset=['GT', 'Patient ID', 'Side'])
gt_df['PatientID'] = gt_df['Patient ID'].astype(int)
gt_df['Side'] = gt_df['Side'].astype(str).str.strip()

# Merge
merged_df = pd.merge(radiomics_df, gt_df[['PatientID', 'Side', 'GT']], on=['PatientID', 'Side'], how='left')

# Fix column names in case of conflict
if 'GT_y' in merged_df.columns:
    merged_df = merged_df.drop(columns=['GT_x'])  # Drop redundant if exists
    merged_df = merged_df.rename(columns={'GT_y': 'GT'})

# Drop rows where GT could not be found
merged_df = merged_df.dropna(subset=['GT'])
merged_df.drop_duplicates(inplace= True)

# Final check
print(merged_df[['ImagePath', 'PatientID', 'Side', 'GT']].head())

merged_df.to_csv(f"Only_radiomics_based_classification/radiomics_features_washu2_p1_143_{position}_label.csv")
