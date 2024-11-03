Aussie Rain Prediction with Decision Tree

Project Description

This project aims to build a predictive model capable of forecasting whether it will rain the following day in Australia. Using historical weather data, the model applies machine learning techniques to predict a binary outcome (RainTomorrow). The project leverages a classic decision tree algorithm (Decision Tree Classifier) for quick and effective prediction based on input features.

Project Functionality

The project includes several key steps to prepare and process the data, ensuring optimal model performance:

	1.	Data Preprocessing:
	•	Handling Missing Values: Missing values are managed using an iterative imputer (IterativeImputer), which estimates missing values to create a complete dataset.
	•	Scaling Numerical Features: Numerical features are scaled with MinMaxScaler to normalize their range, reducing the influence of varying feature scales on the model.
	•	Encoding Categorical Data: Categorical data is processed using one-hot encoding (via OneHotEncoder), enabling the model to effectively handle nominal features.
	2.	Model Training:
	•	The trainModel function trains a Decision Tree Classifier using the preprocessed training data.
	•	The model’s performance is evaluated on training and validation datasets using the ROC AUC score, a key metric for assessing binary classification accuracy.
	3.	Prediction on New Data:
	•	After the model is trained, the preprocess_new_data function prepares new data (e.g., unseen data without RainTomorrow) for prediction by applying the same scaling and encoding used in training.
	•	Predictions on the new data allow for continuous or batch forecasting of rainfall.

Libraries and Tools Used

	•	Python Libraries:
	•	pandas, numpy: For data manipulation and handling
	•	scikit-learn: For model building, preprocessing, and evaluation
	•	imblearn: For oversampling using SMOTE, to address class imbalance
	•	Key Machine Learning Concepts:
	•	Decision Tree Classifier: Used for straightforward, interpretable classification.
	•	ROC AUC Score: Evaluates model accuracy in binary classification tasks.
	•	SMOTE (Synthetic Minority Oversampling Technique): Balances the dataset by creating synthetic samples for the minority class.
Project is deployed here https://aussie-rain-tree2.streamlit.app/
