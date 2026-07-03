import pandas as pd
import numpy as np

# Load datasets
norm_df = pd.read_csv("data/raw/sampled_freeform_full_file.csv")
annotations_df = pd.read_csv("data/raw/annotations.csv")

merged_df = pd.merge(norm_df,annotations_df,on="ID",how="inner")
merged_df


print(f"Merged Dataset Shape: {merged_df.shape}")
print("Duplicate IDs:", merged_df["ID"].duplicated().sum())
merged_df.isnull().sum()
print("Full Dataset:", len(norm_df))
print("Annotations:", len(annotations_df))
print("Merged:", len(merged_df))

merged_df.to_csv("data/processed/moralalign_dataset.csv",index=False)



print(f"Number of Rows    : {merged_df.shape[0]}")
print(f"Number of Columns : {merged_df.shape[1]}")

merged_df.info()
merged_df
print(f"Unique Actions : {merged_df['Action_Valence'].nunique()}")
print(f"Unique Actions : {merged_df['Consequence_Valence'].nunique()}")


merged_df.describe(include='all')
print(merged_df.columns.tolist())


print("CLASS LABEL DISTRIBUTION")
merged_df["text_label"].value_counts()

merged_df[
    [
        "Action_Valence",
        "Consequence_Valence"
    ]
].describe()