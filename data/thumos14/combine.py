import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('/home/ed/video/mm-ZSTAD/src/output/proposals_test_30_st4.csv')

# Start with the first row
combined = [df.iloc[0]]

for _, row in df.iloc[1:].iterrows():
    # If the current row's start time is before the last combined row's end time and the labels are the same,
    # extend the last combined row's end time. Also, update the score if needed.
    if row['t-start'] <= combined[-1]['t-end'] and row['label'] == combined[-1]['label']:
        combined[-1]['t-end'] = max(combined[-1]['t-end'], row['t-end'])
        combined[-1]['score'] = max(combined[-1]['score'], row['score']) # choose the operation according to your needs.
    else:
        # Else start a new row
        combined.append(row)

# Convert the list of combined rows back to a DataFrame
combined_df = pd.DataFrame(combined)

# Write back to CSV
combined_df.to_csv('your_file_combined.csv', index=False)
