#!/usr/bin/env python
# coding: utf-8

# # ***FakeMind-ML4VA Project: Detecting Droughts in Virginia***
# 
# Team FakeMind is composed of three UVA students: Alex Fetea, Kamil Urbanowski, and Tyler Kim. FakeMind's goal is to predict droughts in Virginia using a dataset found online. This will help farmers take better care of their farms by taking preparing ahead of time for possible droughts.
# 
# The link to the datasets can be found below:
# 
# https://resilience.climate.gov/datasets/esri2::us-drought-by-state/explore
# 
# https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
# 
# 
# In general, this notebook will store our code for the ML4VA project. This notebook will also be divided into 8 Steps:
# 
# 1. Big Picture & Setup
# 2. Getting the Data
# 3. Discovering and Visualizing the Data
# 4. Data Cleaning
# 5. Selecting and Training the Models
# 6. Fine Tuning the Model
# 7. Presentation
# 8. Launch
# 

# ## **1-Big Picture & Setup**

# In[15]:


# import the necessary libraries
import sklearn
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

np.random.seed(17)


# ## **2-Getting the Data**

# In[16]:


def load_data(filepath):
    # Read the dataset
    data = pd.read_csv(filepath)

    # Calculate file size in MB
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert bytes to MB

    # Count number of entries and features
    num_entries, num_features = data.shape

    # Count number of categorical features
    num_categorical = sum(data.dtypes == 'object')

    # Check for missing values
    missing_value_exists = data.isnull().values.any()

    # Print the output
    print("File Size: {:.2f} MB".format(file_size))
    print("Number of Entries:", num_entries)
    print("Number of Features:", num_features)
    print("Do Categorical variables exist:", "Yes" if num_categorical > 0 else "No", "({})".format(num_categorical))
    print("Do missing values exist:", "Yes" if missing_value_exists else "No")
    print("\n")

    print(data.info())
    print(data.describe())

    return data


# In[17]:


# loads the data
drought_data = load_data("datasets\merged_weather_drought.csv")


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
drought_data.hist(bins = 50, figsize = (20, 15))
plt.show()


# ## **3-Discovering and Visualizing the Data**

# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming drought_data is your original DataFrame

# Filter for states of interest and create a copy
states_of_interest = ["VA", "MD", "WV", "TN", "KY", "NC", "DC"]
sub_df = drought_data[drought_data['state_abbr'].isin(states_of_interest)].copy()

# Convert 'ddate' to datetime
sub_df['ddate'] = pd.to_datetime(sub_df['ddate'])

# Basic Statistics
print(sub_df.describe())

# Create separate scatter plots for each state
for state in states_of_interest:
    plt.figure(figsize=(10, 6))
    state_data = sub_df[sub_df['state_abbr'] == state]

    plt.scatter(state_data['ddate'], state_data['d0'], label="d0 - Nothing")
    plt.scatter(state_data['ddate'], state_data['d1'], label="d1 - Abnormally Dry")
    plt.scatter(state_data['ddate'], state_data['d2'], label="d2 - Moderate Drought")
    plt.scatter(state_data['ddate'], state_data['d3'], label="d3 - Severe Drought")
    plt.scatter(state_data['ddate'], state_data['d4'], label="d4 - Extreme/Exceptional Drought")

    plt.xlabel('Time')
    plt.ylabel('Drought Level')
    plt.title(f'Drought Levels Over Time in {state}')
    plt.legend()
    plt.show()

# Optionally, create a correlation heatmap as previously described...


# The scatterplots above represent the level of drought for VA and other adjacent states. It seems as if many of the states have a tendency to be abnormally dry but rarely have anything worse.

# ## **4-Data Cleaning**

# Will drop some features since some of it is redundant.

# In[20]:


from sklearn.model_selection import train_test_split

# Define target and feature columns
output_list = ["nothing", "d0", "d1", "d2", "d3", "d4"]
feature_cols = [col for col in drought_data.columns if col not in output_list and col not in ['D0_D4', 'D1_D4', 'D2_D4', 'D3_D4']]

# Split the data into training and testing sets
train_set, test_set = train_test_split(drought_data, test_size=0.2, random_state=17)

# Separate features and target for training data
X_train = train_set[feature_cols]
y_train = train_set[output_list]

print(y_train.head())
print(X_train.head())


# In[21]:


# Test distributions
print(X_train["state_abbr"].value_counts() / len(X_train))
print(drought_data["state_abbr"].value_counts() / len(drought_data))


# In[22]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Define the pipeline for numeric and categorical attributes
num_attribs = list(X_train.select_dtypes(include=np.number))
cat_attribs = list(X_train.select_dtypes(exclude=np.number))

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Impute with mean
    ("std_scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


# In[23]:


# Apply the full pipeline to the training data
X_train_prepared = full_pipeline.fit_transform(X_train).toarray()

print(y_train)

# Converting to one-hot encoded format
y_train_one_hot = np.zeros_like(y_train.values)
y_train_one_hot[np.arange(len(y_train.values)), y_train.values.argmax(axis=1)] = 1

# Convert back to DataFrame
y_train_one_hot_df = pd.DataFrame(y_train_one_hot, columns=y_train.columns)

print(y_train_one_hot_df.head())


# ## **5-Selecting and Training the Models**

# In[24]:


import tensorflow as tf
from tensorflow import keras

num_samples = X_train_prepared.shape[0]
num_features = X_train_prepared.shape[1]
num_outputs = y_train_one_hot_df.shape[1]
timesteps = X_train_prepared.size // (num_samples * num_features)

# Reshape the data
X_train_prepared_reshaped = X_train_prepared.reshape((num_samples, timesteps, num_features))


# Model building function for RNN classification
def build_rnn_classification_model(input_shape, n_units=[50, 50], num_outputs=num_outputs, dropout_rate=0.3):
    model = keras.models.Sequential()
    
    # Input layer
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # LSTM layers
    for i, n_unit in enumerate(n_units):
        return_sequences = i < len(n_units) - 1
        model.add(keras.layers.LSTM(n_unit, return_sequences=return_sequences))
        model.add(keras.layers.Dropout(dropout_rate))

    # Output layer for multi-class classification
    model.add(keras.layers.Dense(num_outputs, activation='softmax'))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Building and compiling the RNN model
input_shape = (timesteps, num_features)
rnn_classification_model = build_rnn_classification_model(input_shape)

# Model summary
rnn_classification_model.summary()


# In[25]:


# Train the model
history = rnn_classification_model.fit(
    X_train_prepared_reshaped,
    y_train_one_hot_df,
    epochs=20,
    validation_split=0.1
)


# In[44]:


import numpy as np
import pandas as pd
from tensorflow import keras

# Assuming X_train_prepared and y_train are already defined and preprocessed


# Number of classes - adjust this based on your dataset
num_classes = y_train_one_hot_df.shape[1]

# Model building function for multi-class logistic regression
def build_multiclass_logistic_regression_model(input_shape, learning_rate=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(num_classes, input_shape=input_shape, activation='softmax'))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Building and compiling the logistic regression model
num_features = X_train_prepared.shape[1]
logistic_regression_model = build_multiclass_logistic_regression_model((num_features,))

# Model summary
logistic_regression_model.summary()

# Train the model
history = logistic_regression_model.fit(
    X_train_prepared,
    y_train_one_hot_df,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)


# In[27]:


# Train the model
history = logistic_regression_model.fit(
    X_train_prepared,
    y_train,
    epochs=20,
    validation_split=0.1
)


# In[28]:


keras.backend.clear_session()
np.random.seed(42)


# ## **6-Fine Tuning the Model**

# In[ ]:





# ## **7-Presentation**

# In[ ]:





# ## **8-Launch**

# In[ ]:




