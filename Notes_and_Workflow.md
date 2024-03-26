# Data Handling and ETL Process

**Current:** Mix of database (for initial data); parquet; and other filetypes. Nested folders creating complex filepaths. Too many different versions of the same data.

**MVP Solution:**
* Use only parquet files
* Focus on `ArrivalDate` for Datetime index (DTI)
    * `BookingDate` will be AAB
* Ensure all notebooks are fully functional

**AAB Goals:**
* Copy workflow for `BookingDate` as DTI.
* Proper database setup and maintenance.
* Normalization of database tables.
* Rework workflow to leverage database strengths and functionality.
* Restructure full repository for clarity and organization.

# Feature Selection - Known in Advance

Identify and subset features most likely to be known before/at booking/arrival.

Ask the question, *Which features would I know during a revenue management call?*

# Auxillary Models

## Clustering

Using a clustering algorithm to add new features to your dataset is an effective way to encapsulate complex patterns or relationships within your data into new, meaningful features. These new features can enhance model performance by introducing new dimensions of information that were not explicitly present before. Here’s a step-by-step approach on how to use clustering for feature engineering:

### 1. **Choose a Clustering Algorithm**

The choice of clustering algorithm can depend on the nature of your data and the specific patterns you're aiming to capture. Common choices include:

- **K-Means Clustering:** Good for capturing spherical clusters in feature space.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Useful for identifying clusters of varying shapes and sizes, and for data with noise.
- **Hierarchical Clustering:** Useful for when you want a hierarchy of clusters.

### 2. **Preprocess the Data**

Clustering algorithms usually require numerical input features and can be sensitive to the scale of the data:

- **Normalization/Standardization:** Apply normalization (scaling features to a [0, 1] range) or standardization (scaling features to have a mean of 0 and a variance of 1) to ensure that all features contribute equally to the distance calculations used in most clustering algorithms.
- **Handling Missing Values:** Ensure that missing values are imputed or handled appropriately, as clustering algorithms generally do not support missing values.

### 3. **Perform Clustering**

Perform clustering on either the entire dataset or a subset of features that you believe encapsulate interesting patterns or groupings. The output will be a cluster label for each data point.

```python
from sklearn.cluster import KMeans

# Assuming you've selected K-Means clustering and determined the number of clusters (k)
k = 5
clustering_model = KMeans(n_clusters=k, random_state=42)

# Fit the model on your data (excluding the target variable if it's supervised learning)
cluster_labels = clustering_model.fit_predict(X)
```

### 4. **Add Cluster Labels as New Features**

The cluster labels can be added back to your dataset as a new feature, which might capture non-linear relationships and complex patterns in a way that's useful for predictive modeling.

```python
# Add the cluster labels as a new feature to your DataFrame
df['cluster_label'] = cluster_labels
```

### 5. **Optional: Distance to Cluster Centroid**

For algorithms like K-Means, another informative feature can be the distance of each point to its cluster centroid, which provides information about how centrally located or peripheral a point is within its cluster.

```python
import numpy as np

# Calculate distances to cluster centroid
distances = np.min(clustering_model.transform(X), axis=1)
df['distance_to_centroid'] = distances
```

### Considerations and Tips

- **Feature Selection for Clustering:** Depending on your data, you might want to perform clustering based on a subset of features that are particularly relevant for capturing the patterns of interest.
- **Cluster Validity:** Assess the validity and stability of the clusters formed, using metrics like silhouette score, to ensure they are meaningful and consistent.
- **Model Evaluation:** Always validate the impact of adding these new features on the performance of your predictive models through cross-validation or a hold-out validation set.

Adding clustering-based features can introduce a rich layer of information to your models, potentially capturing complex patterns that were not utilized previously. However, it's also essential to monitor for overfitting and ensure the clustering process itself is robust and meaningful.

### KMeans Clustering - Numeric 

Focusing solely on the numeric features simplifies the choice of clustering algorithms for your dataset of 120,000 rows, as you no longer need to consider the complexity of handling mixed data types. This allows you to leverage the strengths of algorithms that are well-suited to large, numeric datasets. Here are the most suitable options:

#### 1. **K-Means**

- **Scalability:** Excellent. K-Means is efficient for large datasets, particularly with the Mini-Batch K-Means variant, which is designed to be more scalable by processing subsets of the data in each iteration.
  
- **Implementation:** Use `MiniBatchKMeans` from Scikit-learn for a more scalable version of K-Means. Remember to standardize your numeric features since K-Means is sensitive to the scale of the data.

#### 2. **DBSCAN**

- **Scalability:** Good with optimizations. DBSCAN can handle large datasets, but its performance might degrade as the size increases. Its complexity is primarily based on the number of rows and the efficiency of the nearest neighbor search.
  
- **Implementation:** Consider using optimized libraries like HDBSCAN (an extension of DBSCAN available in a separate Python package) that can handle larger datasets more efficiently and offers automatic parameter tuning.

#### 3. **Hierarchical Clustering**

- **Scalability:** Poor. Hierarchical clustering is computationally intensive for large datasets due to its \(O(n^2)\) complexity in both time and space, making it less suitable for 120,000 rows.

#### 4. **Mean Shift**

- **Scalability:** Poor. Like hierarchical clustering, mean shift is not typically recommended for large datasets due to its high computational cost.

#### 5. **Spectral Clustering**

- **Scalability:** Poor for large datasets. Spectral clustering involves eigenvalue decomposition of the similarity matrix, which becomes computationally expensive as the dataset grows.

#### Scalable Clustering Recommendations:

Given the focus on numeric features and considering the dataset's size:

- **Mini-Batch K-Means** is highly recommended due to its scalability and efficiency. It's particularly suitable if you can predefine the number of clusters or use methods like the elbow method or silhouette analysis to determine an optimal cluster count.
  
- **DBSCAN or HDBSCAN** might be suitable if your data includes noise or if you're looking for clusters with arbitrary shapes. HDBSCAN, in particular, offers advantages in handling variable density clusters and automatically determining the number of clusters, which can be beneficial for exploratory data analysis.

#### Steps for Mini-Batch K-Means:

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# Assuming 'df_numeric' is your DataFrame containing only numeric features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Optimal number of clusters (k) needs to be determined
k = 5  # Example placeholder value
mini_batch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
clusters = mini_batch_kmeans.fit_predict(df_scaled)

# Add cluster labels to your DataFrame
df_numeric['cluster'] = clusters
```

#### Conclusion:

Focusing on numeric features allows you to utilize scalable and efficient clustering algorithms like Mini-Batch K-Means and potentially DBSCAN or HDBSCAN, depending on your specific needs and the characteristics of your data. Always ensure proper preprocessing like feature scaling to improve clustering outcomes and validate the chosen number of clusters through silhouette scores or similar metrics when using methods like K-Means.


## Isolation Forest

Using an Isolation Forest for anomaly detection can be applied in different ways depending on your goals, especially concerning training data, test data, endogenous variables (variables of interest in your model, typically the target variable in supervised learning), and exogenous variables (independent variables or features). Here's how you might consider applying it:

### Using Isolation Forest on Training Data

- **Objective:** The primary objective here is to clean your training dataset by removing outliers that could potentially skew the model training process.
- **Variables:** It's typically applied to the exogenous variables (features) because you're interested in identifying instances with unusual combinations of feature values that do not conform to the normal patterns observed in the data.
- **Process:**
    - Fit the Isolation Forest on the training dataset.
    - Identify and optionally remove anomalies.
    - Proceed with model training on the cleaned dataset.

### Using Isolation Forest on Test Data

- **Objective:** While less common for pre-processing, applying an Isolation Forest to test data could be part of a post-modeling pipeline, especially in use cases like fraud detection where identifying outliers in new data is continuously relevant.
- **Variables:** Similar to the training data, the focus would typically be on the exogenous variables. However, the intention here is not to clean the data but to flag potentially anomalous instances for further inspection.
- **Process:**
    - The Isolation Forest model should be fitted only to the training data to avoid information leakage.
    - Apply the fitted model to the test data to identify anomalies.

### Using Isolation Forest on Endogenous Variables

- **Objective:** Though less common, you might use an Isolation Forest on the endogenous variable if your goal is to identify outliers in the target variable itself, separate from any modeling process. This can be useful in data cleaning or exploratory analysis stages.
- **Process:**
    - Fit the Isolation Forest directly on the endogenous variable if looking for outliers in the target distribution before proceeding with other analyses.

### Best Practices

1. **Primary Use on Features (Exogenous Variables):** The typical use of Isolation Forest is on the feature set to clean the training data or to detect anomalies in operational data. This helps in building more robust predictive models and identifying outliers in ongoing data feeds.
   
2. **Consider the Context:** Whether to apply it to training or test data (or both) depends on your specific goals. For data cleaning and model training, apply it to training data. For operational anomaly detection, apply it to new, unseen data.

3. **Caution with Test Data:** Be cautious about removing identified anomalies from the test data before model evaluation, as this could lead to overly optimistic performance metrics. Anomalies in test data can provide valuable insights into how the model might perform in the real world, where anomalous instances can occur.

4. **Evaluation:** After identifying anomalies, evaluate them in the context of your specific problem. Sometimes, what is considered an anomaly by the algorithm might be a valuable edge case for the business or require a different kind of handling or analysis.

By thoughtfully applying Isolation Forest to your data considering these aspects, you can enhance both the quality of your data for modeling and the insights you derive from your analysis.

## Kalman Filter

> See ChatGPT chat: https://chat.openai.com/share/6864f390-80fc-423d-91f6-88a9b8cd8f1e


# Inclusion of Error and Predictive Metrics

Incorporating error metrics or predictive metrics from auxiliary models as features in an advanced dataset is an unconventional approach, but it can provide unique insights in certain contexts. Whether or not to include these metrics depends on your specific problem, the nature of your data, and what you're trying to predict. Here are some considerations and potential scenarios where such an approach might make sense:

## Potential Benefits

1. **Error Patterns as Features:** Sometimes, the error patterns from auxiliary models might carry information about the data's structure or underlying phenomena that are not captured by other features. For instance, if certain instances consistently yield higher prediction errors, this might indicate a subgroup with distinct characteristics.

2. **Model Confidence:** Predictive metrics from auxiliary models, such as probability estimates for classification tasks, can serve as a proxy for the model's confidence in its predictions. These metrics can provide additional context that could be useful for the main model.

## How to Implement

- **Generate Error Metrics:** After training your auxiliary models, calculate the error metrics (e.g., MSE, RMSE, MAE for regression tasks; log loss, or Brier score for classification) or predictive metrics (e.g., predicted probabilities) for each instance in your dataset.
  
- **Incorporate as Features:** Add these metrics as new features to your dataset. This involves appending the error or confidence metrics from the auxiliary models to your feature set before training your main model.

## Considerations

- **Risk of Overfitting:** Adding these metrics as features could increase the risk of overfitting, especially if the auxiliary models are highly overfitted to the training data. It’s crucial to apply regularization techniques and perform thorough cross-validation.

- **Interpretability:** Including error metrics or predictive metrics as features can complicate the interpretation of your model. It might not be immediately clear how these features interact with others or their overall impact on the predictions.

- **Dependency on Auxiliary Models:** Your main model's performance and predictions will now depend on the auxiliary models' performance. Any changes or updates to the auxiliary models could impact your main model's behavior.

- **Model Complexity:** This approach adds complexity to your modeling process. The additional preprocessing steps, dependency management, and the need for careful validation can complicate model development and maintenance.

## Example Use Cases

- **Anomaly Detection:** If auxiliary models are used to detect anomalies, the error metrics might indicate instances that are difficult to model, potentially pointing to outliers or anomalies.
  
- **Ensemble Techniques:** In ensemble methods, confidence scores from individual models are often used to weigh predictions. Similarly, incorporating predictive metrics from auxiliary models could be seen as a form of ensemble learning.

## Final Thoughts

While innovative, using error metrics or predictive metrics from auxiliary models as features in your dataset is a strategy that requires careful consideration and rigorous validation. It might offer benefits in specific scenarios, particularly when conventional features do not fully capture the complexity of the data or the task at hand. However, it's essential to weigh these benefits against the potential risks and complexities introduced.