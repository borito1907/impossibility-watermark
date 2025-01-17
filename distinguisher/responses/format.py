import pandas as pd

df1 = pd.read_csv('../../jason_gens/adaptive_gens_1/watermarked_texts.csv')
df2 = pd.read_csv('../../jason_gens/adaptive_gens_2/watermarked_texts.csv')

df_combined = pd.concat([df1, df2]).sort_values(by='entropy')
cols = df_combined.columns.tolist()
cols[0], cols[1] = cols[1], cols[0]
df_combined = df_combined[cols]
df_combined.to_csv('adaptive_responses.csv', index=False)

df3 = pd.read_csv('../../jason_gens/umd_gens_1/watermarked_texts.csv')
df3 = df3.drop_duplicates(subset='entropy', keep='last')
df4 = pd.read_csv('../../jason_gens/umd_gens_2/watermarked_texts.csv')

df_combined2 = pd.concat([df3, df4]).sort_values(by='entropy')
df_combined2 = df_combined2.merge(df_combined[['entropy', 'prompt']].drop_duplicates(subset = 'entropy'), on='entropy', how='left')
columns = ['prompt'] + [col for col in df_combined2.columns if col != 'prompt']
df_combined2 = df_combined2[columns]
df_combined2.to_csv('umd_responses.csv', index=False)