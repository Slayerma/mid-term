# mid-term credit card fraud detection


Credit card fraud is a significant concern for financial institutions and customers. This code presents a machine learning approach to detect credit card fraud using the Random Forest algorithm.

Importing Libraries:

We import the necessary libraries: pandas, matplotlib.pyplot, seaborn, RandomForestClassifier, classification_report, confusion_matrix, roc_curve, roc_auc_score, and train_test_split. These libraries provide essential functionalities for data analysis, model training, evaluation, and visualization.

Loading the Dataset:

We load the "creditcard.csv" dataset using pandas' read_csv() function. This dataset contains credit card transaction data, including features like transaction amount, timestamp, and a target variable indicating fraud or legitimacy.

Exploring and Preprocessing the Data:

We check for missing values in the dataset using isnull().sum(). This step ensures the data quality and helps us handle any missing values if present.

Splitting the Data:

We split the dataset into features (X) and labels (y). Features represent the transaction details, while labels indicate fraud (1) or legitimacy (0). We use the train_test_split() function to divide the data into training and testing sets.

Training the Random Forest Model:

We initialize a Random Forest classifier using RandomForestClassifier(). The model is trained on the training data using fit().

Predicting and Evaluating:

We make predictions on the testing set using the trained model and predict(). The model assigns predicted class labels (fraud or legitimacy) to the transactions. We evaluate the model's performance by printing the classification report, confusion matrix, and ROC AUC score using classification_report(), confusion_matrix(), and roc_auc_score().

Data Visualization - Confusion Matrix:

We create a confusion matrix visualization using sns.heatmap(). This heatmap represents the true and predicted labels of the transactions. The annotation and color intensity indicate the number of correctly and incorrectly classified instances.

Data Visualization - ROC Curve:

We plot the Receiver Operating Characteristic (ROC) curve using roc_curve(). The ROC curve illustrates the trade-off between true positive rate and false positive rate. This visualization helps assess the model's performance and determine an optimal threshold.

Data Visualization - Count Plot:

We create a count plot using seaborn's countplot() function. The count plot represents the distribution of credit card fraud cases. The x-axis represents the class labels (0: Legitimate, 1: Fraud), and the y-axis represents the count.

Data Visualization - Annotating Fraud Cases:

We annotate the count plot to display the number of fraud cases. The annotation is placed near the height of the fraud bar on the plot. This provides a clear visual representation of the number of fraud cases

Conclusion:

The presented code demonstrates a credit card fraud detection solution using the Random Forest algorithm. By training the model on credit card transaction data, we can effectively identify fraudulent transactions. The evaluation metrics and visualizations provide insights into the model's performance and assist in decision-making.


