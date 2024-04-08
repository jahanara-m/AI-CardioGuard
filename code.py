# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load the dataset
df = pd.read_csv("AI_Dataset.csv")

# Remove records with null values
df.dropna(inplace=True)

# Reset the indices to avoid potential issues
df = df.reset_index(drop=True)

# Standard Scalar (Z-score normalization)
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Outlier Detection and Removal (using z-score method)
z_scores = (df - df.mean()) / df.std()
outliers = (z_scores > 3) | (z_scores < -3)
df_cleaned = df[~outliers.any(axis=1)]

# One-Hot Encoding for categorical variables
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols)

# Extracting features and target variable
X = df_encoded.drop('target', axis=1)  # Features
y = df_encoded['target']  # Target variable

# CNN Model
def create_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # Output layer with softmax for binary classification

    # Compile the model with a low learning rate
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Define the number of folds for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}:")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Reshape data for CNN
    X_train_cnn = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convert data types to float32
    X_train_cnn = X_train_cnn.astype('float32')
    X_test_cnn = X_test_cnn.astype('float32')

    # Check for NaN or infinite values and replace them with zeros
    X_train_cnn = np.nan_to_num(X_train_cnn)
    X_test_cnn = np.nan_to_num(X_test_cnn)

    # Create and train the model
    model = create_model()
    history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test_cnn, y_test), verbose=1)

    # Evaluate the model
    y_pred_cnn = np.argmax(model.predict(X_test_cnn), axis=-1)
    accuracy_scores.append(accuracy_score(y_test, y_pred_cnn))
    precision_scores.append(precision_score(y_test, y_pred_cnn))
    recall_scores.append(recall_score(y_test, y_pred_cnn))
    f1_scores.append(f1_score(y_test, y_pred_cnn))

    # Print evaluation metrics for each fold
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy_scores[-1]:.3f}")
    print(f"Precision: {precision_scores[-1]:.3f}")
    print(f"Recall: {recall_scores[-1]:.3f}")
    print(f"F1-score: {f1_scores[-1]:.3f}")
    print("-" * 50)

# Calculate average evaluation metrics across all folds
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print("Average Cross-Validation Metrics:")
print(f"Accuracy: {average_accuracy:.3f}")
print(f"Precision: {average_precision:.3f}")
print(f"Recall: {average_recall:.3f}")
print(f"F1-score: {average_f1:.3f}")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Flatten, RepeatVector, Permute, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import keras.backend as K

# Load the dataset
df = pd.read_csv("AI_Dataset.csv")

# Remove records with null values
df.dropna(inplace=True)

# Reset the indices to avoid potential issues
df = df.reset_index(drop=True)

# Standard Scalar (Z-score normalization)
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Outlier Detection and Removal (using z-score method)
z_scores = (df - df.mean()) / df.std()
outliers = (z_scores > 3) | (z_scores < -3)
df_cleaned = df[~outliers.any(axis=1)]

# One-Hot Encoding for categorical variables
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols)

# Extracting features and target variable
X = df_encoded.drop('target', axis=1).values.astype(np.float32)  # Convert X to float32
y = df_encoded['target'].values.astype(np.float32)  # Convert y to float32

# Define the LSTM model with attention mechanism
def lstm_attention_model(input_shape, dropout_rate=0.2):
    num_features = input_shape[0]
    num_time_steps = 1  # We set the number of time steps to 1

    inputs = Input(shape=(num_time_steps, num_features))  # Adjust input shape to include time steps and features

    # LSTM layer with dropout and batch normalization
    lstm_out = LSTM(256, return_sequences=True)(inputs)  # Increased LSTM units
    lstm_out = Dropout(dropout_rate)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = Dense(num_time_steps, activation='softmax')(attention)
    attention = RepeatVector(num_features)(attention)
    attention = Permute((2, 1))(attention)

    # Applying attention
    attention_out = Concatenate(axis=-1)([lstm_out, attention])

    attention_out = LSTM(128)(attention_out)  # Increased LSTM units

    # Output layer
    outputs = Dense(1, activation='sigmoid')(attention_out)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the weighted binary cross-entropy loss function with target repeat prediction
def weighted_binary_crossentropy(y_true, y_pred, weights):
    # Apply weights based on the true labels
    weighted_loss = weights[0] * (y_true * K.log(y_pred + K.epsilon())) + weights[1] * ((1 - y_true) * K.log(1 - y_pred + K.epsilon()))
    return -K.mean(weighted_loss, axis=-1)

# Define the number of folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}:")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Instantiate the model
    model = lstm_attention_model((X_train.shape[1],), dropout_rate=0.3)  # Adjusted dropout rate

    # Compile the model with weighted loss
    weights = [2, 1]  # Define weights for binary_crossentropy
    model.compile(optimizer=Adam(learning_rate=0.001), loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, weights), metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train, epochs=100, batch_size=32, validation_data=(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), y_test))

    # Evaluate the model
    y_pred = model.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
    y_pred_class = (y_pred > 0.5).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    # Print evaluation metrics for each fold
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("-" * 50)

    # Append evaluation metrics to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Calculate average evaluation metrics across all folds
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print("Average Cross-Validation Metrics:")
print(f"Accuracy: {average_accuracy:.3f}")
print(f"Precision: {average_precision:.3f}")
print(f"Recall: {average_recall:.3f}")
print(f"F1-score: {average_f1:.3f}")

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.optimizers import Adam

# Load the dataset
df = pd.read_csv("AI_Dataset.csv")

# Remove records with null values
df.dropna(inplace=True)

# Reset the indices to avoid potential issues
df = df.reset_index(drop=True)

# Standard Scalar (Z-score normalization)
scaler = StandardScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Outlier Detection and Removal (using z-score method)
z_scores = (df - df.mean()) / df.std()
outliers = (z_scores > 3) | (z_scores < -3)
df_cleaned = df[~outliers.any(axis=1)]

# One-Hot Encoding for categorical variables
categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols)

# Extracting features and target variable
X = df_encoded.drop('target', axis=1)  # Features
y = df_encoded['target']  # Target variable

# CNN Model
def create_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid for binary classification

    # Compile the model with a low learning rate
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Define the number of folds for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}:")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Reshape data for CNN
    X_train_cnn = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    # Convert data types to float32
    X_train_cnn = X_train_cnn.astype('float32')
    X_test_cnn = X_test_cnn.astype('float32')

    # Check for NaN or infinite values and replace them with zeros
    X_train_cnn = np.nan_to_num(X_train_cnn)
    X_test_cnn = np.nan_to_num(X_test_cnn)

    # Create and train the model
    model = create_model()
    history = model.fit(X_train_cnn, y_train, epochs=100, batch_size=8,
                        validation_data=(X_test_cnn, y_test), verbose=1)

    # Evaluate the model
    y_pred_cnn = np.round(model.predict(X_test_cnn)).astype(int)
    accuracy_scores.append(accuracy_score(y_test, y_pred_cnn))
    precision_scores.append(precision_score(y_test, y_pred_cnn))
    recall_scores.append(recall_score(y_test, y_pred_cnn))
    f1_scores.append(f1_score(y_test, y_pred_cnn))

    # Print evaluation metrics for each fold
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy_scores[-1]:.3f}")
    print(f"Precision: {precision_scores[-1]:.3f}")
    print(f"Recall: {recall_scores[-1]:.3f}")
    print(f"F1-score: {f1_scores[-1]:.3f}")
    print("-" * 50)

# Calculate average evaluation metrics across all folds
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

print("Average Cross-Validation Metrics:")
print(f"Accuracy: {average_accuracy:.3f}")
print(f"Precision: {average_precision:.3f}")
print(f"Recall: {average_recall:.3f}")
print(f"F1-score: {average_f1:.3f}")