import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the dataset
df = pd.read_csv("kidney_disease.csv")

# Define columns to retain
columns_to_retain = ['sg', 'al', 'sc', 'hemo', 'pcv', 'htn', 'classification', 'wc', 'rc']

# Check if all columns in columns_to_retain exist in the dataset
missing_columns = [col for col in columns_to_retain if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following columns are missing in the dataset: {missing_columns}")

# Preprocess the data
df = df[columns_to_retain].dropna()

# Encode categorical variables
for column in df.columns:
    if df[column].dtype == object:  # Use 'object' instead of 'np.object'
        df[column] = LabelEncoder().fit_transform(df[column])

# Split data into features and target
X = df.drop(['classification'], axis=1)
y = df['classification']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for LSTM input (samples, timesteps, features)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', accuracy)

# Make predictions
predictions = model.predict(X_test)
predictions = ['NotCKD' if pred >= 0.5 else 'CKD' for pred in predictions]
print('Original:{0}'.format(",".join(str(x)for x in y_test)))
print('Predicted:{0}'.format(",".join(str(x)for x in predictions)))
print('Predictions:', predictions)
