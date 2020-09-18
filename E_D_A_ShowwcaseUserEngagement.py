#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 22:48:29 2020

@author: akshay9
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

showwcase_user_engagement = pd.read_csv('/Users/akshay9/Desktop/Showwcase/showwcase_data.csv')
aggregated_user_engagement = showwcase_user_engagement.groupby(['customer_id']).sum().sort_values('session_duration')

aggregated_user_engagement = aggregated_user_engagement.rename(columns={'session_projects_added':'Total_Projects_Added' ,'session_likes_given':'Total_Likes_Given', 'session_comments_given':'Total_Comments_Given','bugs_in_session':'Total_bugs_faced_by_User','session_duration':'Total_Duration' })


aggregated_user_engagement.drop(['session_id','projects_added','likes_given','comment_given','bug_occured','inactive_status'],axis = 1, inplace=True)
aggregated_user_engagement = aggregated_user_engagement.reset_index(level=0)

print(aggregated_user_engagement[['customer_id','Total_Projects_Added','Total_bugs_faced_by_User','Total_Duration']])

X = aggregated_user_engagement.iloc[:,[0,1]].values



#print(aggregated_user_engagement[['session_duration','session_projects_added','bugs_in_session']])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Optimal Number of Clusters:2


kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of user enganement of Showwcase')
plt.xlabel('Customer ID')
plt.ylabel('Projects Uploaded')
plt.legend()
plt.show()
