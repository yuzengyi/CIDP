# Re-import necessary libraries and load the data again
import pandas as pd
from sklearn.cluster import KMeans

# Load the previously loaded Excel file
file_path = 'newdata/data_first.xlsx'
data = pd.read_excel(file_path)

# Extract 'Lev' column for clustering, ensuring to drop missing values
lev_data = data[['Lev']].dropna()

# Perform K-Means clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
data['Y'] = kmeans.fit_predict(lev_data)

# Create a mapping to ensure the highest Lev values get the highest Y category
# Sort by 'Lev' and get the mapping of old cluster labels to new labels
sorted_clusters = data[['Lev', 'Y']].sort_values(by='Lev').groupby('Y').mean().sort_values(by='Lev').index
new_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}

# Apply the new labels to the 'Y' column
data['Y'] = data['Y'].map(new_labels)

# Save the updated dataframe to a new Excel file
output_path = 'new_data_with_clusters_sorted.xlsx'
data.to_excel(output_path, index=False)
