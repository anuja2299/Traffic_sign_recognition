from skimage.feature import hog
from skimage import io, color
import csv
import os
import cv2
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

def extract_features_single(image):
    gray_image = color.rgb2gray(image)
    new_image_resized = cv2.resize(gray_image, (32,32))
    features = hog(new_image_resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features


# Directory containing the images for training
data_dir = '/content/drive/MyDrive/ML project/dataset'
categories = os.listdir(data_dir)  # Assuming each subdirectory corresponds to a category

# Collecting images and corresponding labels
images = []
labels = []

for category in categories:
    category_dir = os.path.join(data_dir, category)
    for file_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, file_name)
        image = io.imread(img_path)
        if image.shape[2] == 4:
          image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        features = extract_features_single(image)
        images.append(features)
        labels.append(category)

# Save features and labels to a CSV file
csv_file = 'hog_features.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow([f'Feature_{i}' for i in range(len(images[0]))] + ['Label'])

    # Write rows of features and labels
    for i in range(len(images)):
        writer.writerow(images[i].tolist() + [labels[i]])
csv_file


# Load HOG features and labels from the CSV file
csv_file = 'hog_features.csv'
data = pd.read_csv(csv_file)

# Separating features and labels
features = data.drop('Label', axis=1)
labels = data['Label']



# Splitting the dataset into training and testing sets
X_train, X_test, y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


#SVM CLASSIFIER
# Training the SVM classifier
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)

# Making predictions
y_pred= clf.predict(X_test)


# Calculating accuracy
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))



#PREDICTION OF NEW DATA WITH SVM
 #Path to the folder containing images for prediction
folder_path = '/content/drive/MyDrive/ML project/data set for prediction'


# Predict labels for images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg' or '.jpeg' or '.png'):  # Adjust file extensions accordingly
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Extract features from the image
        image_features = extract_features_single(image)

        # Reshape features to match the format expected by the classifier
        image_features = np.array(image_features).reshape(1, -1)

        # Predict label using the trained model
        predicted_label = clf.predict(image_features)

        print(f"Image: {filename}, Predicted Label: {predicted_label}")



import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'hog_features.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Assume the last column is the target/label column
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Binarize the labels
y_bin = label_binarize(y, classes=y.unique())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Train the SVM classifier using OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
clf.fit(X_train, y_train)

# Predict probabilities for each class on the test set
y_pred_proba = clf.predict_proba(X_test)

# Initialize variables for precision and recall
precision = dict()
recall = dict()

# Compute precision and recall for each class
for i in range(len(y.unique())):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])

# Calculate average precision and average recall
average_precision = average_precision_score(y_test, y_pred_proba, average='weighted')

# Plot precision-recall curves for each class
plt.figure(figsize=(10, 7))

for i in range(len(y.unique())):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

# Plot the average precision-recall curve
plt.plot(recall[i], precision[i], lw=2, color='black', label='Average Precision-Recall Curve')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()




from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ... (your prediction code)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
num_classes = len(np.unique(np.concatenate((y_test, y_pred))))
# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Create KNN classifier with 5 neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier
knn_classifier.fit(X_train, y_train)

# Optional: Make predictions
y_pred = knn_classifier.predict(X_test)

# Optional: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Path to the folder containing images for prediction
folder_path = '/content/drive/MyDrive/ML project/data set for prediction'

# Predict labels for images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg' or '.jpeg' or '.png'):  # Adjust file extensions accordingly
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Extract features from the image
        image_features = extract_features_single(image)

        # Reshape features to match the format expected by the classifier
        image_features = np.array(image_features).reshape(1, -1)

        # Predict label using the trained model
        predicted_label = knn_classifier.predict(image_features)

        print(f"Image: {filename}, Predicted Label: {predicted_label}")


