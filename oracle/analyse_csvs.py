import pandas as pd


def get_res(standard_csv = '/home/u1411251/tools/my_scripts/res_standard.csv', oracle_csv = '/home/u1411251/tools/my_scripts/res_oracle.csv'):
    df1 = pd.read_csv(standard_csv)
    df2 = pd.read_csv(oracle_csv)

    # Apply your conditions
    filtered_df1 = df1[df1['result1'] == 'fp']

    filtered_df2 = df2[df2['result1'] == 'tn']

    merged_df = pd.merge(filtered_df1, filtered_df2,
        on=['netname', 'image_index', 'epsilon'],  # Match on these two columns
        how='inner'                  # Only keep matching rows
    )

    merged_df.to_csv('fp_to_tn.csv', index=False)

get_res()
    