import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Load the Excel file
file_path = 'data_first.xlsx'
data = pd.read_excel(file_path)
# Set the font to Times New Roman and font size to 14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Extract 'Lev' column for clustering
lev_data = data[['Lev']].dropna()

# Function to calculate the Sum of Squared Distances (SSD) for a range of k values
def calculate_ssd(data, k_range):
    ssd = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        ssd.append(kmeans.inertia_)
    return ssd

# Range of k values to test
k_values = range(1, 11)
ssd_values = calculate_ssd(lev_data, k_values)

# Plot the elbow graph to determine the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(k_values, ssd_values, marker='o')
plt.title('Elbow Method for Determining Optimal k in K-Means Clustering')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.grid(True)
plt.show()
