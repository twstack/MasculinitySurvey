import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import warnings


def warn(*args, **kwargs):
    pass

warnings.warn = warn

survey = pd.read_csv("masculinity.csv")

cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
               "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
               "q0007_0010", "q0007_0011"]

for col in cols_to_map:
    survey[col] = survey[col].map({
        "Never, and not open to it": 0,
        "Never, but open to it": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Often": 4
    })

plt.scatter(survey["q0007_0001"], survey["q0007_0002"], alpha=0.1)
plt.xlabel("Ask a friend for professional advice")
plt.ylabel("Ask a friend for personal advice")
plt.show()

# 7.1 - 7.4 = not exemplary of stereotypical masculine traits
# 7.5, 7.8, 7.9 = more exemplary of stereotypical masculine traits
# Questions found in the survey pdf

rows_to_cluster = survey.dropna(subset=["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"])
classifier = KMeans(n_clusters=2)
classifier.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])
print(classifier.cluster_centers_, "\n")

# "When we look at the two clusters, the first four numbers represent the traditionally feminine activities and the last
# three represent the traditionally masculine activities. If the data points separated into a feminine cluster and a
# masculine cluster, we would expect to see one cluster to have high values for the first four numbers and the other
# cluster to have high values for the last three numbers.\nInstead, the first cluster has a higher value in every
# feature. Since a higher number means the person was more likely to "often" do something, the clusters seem to
# represent 'people who do things' and 'people who don't do things'."

cluster_zero_indices = []
cluster_one_indices = []
for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_zero_indices.append(i)
    elif classifier.labels_[i] == 1:
        cluster_one_indices.append(i)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

print(cluster_zero_df['educ4'].value_counts()/len(cluster_zero_df), "\n")
print(cluster_one_df['educ4'].value_counts()/len(cluster_one_df), "\n")

print("----------------------------------------------------------", "\n")

print(cluster_zero_df['orientation'].value_counts()/len(cluster_zero_df), "\n")
print(cluster_one_df['orientation'].value_counts()/len(cluster_one_df), "\n")

print("----------------------------------------------------------", "\n")

print(cluster_zero_df['age3'].value_counts()/len(cluster_zero_df), "\n")
print(cluster_one_df['age3'].value_counts()/len(cluster_one_df), "\n")

print("----------------------------------------------------------", "\n")

print(cluster_zero_df['racethn4'].value_counts()/len(cluster_zero_df), "\n")
print(cluster_one_df['racethn4'].value_counts()/len(cluster_one_df), "\n")

# race seems to factor into any answer differences the least
# sexuality and schooling seem to factor into any answer differences the most
