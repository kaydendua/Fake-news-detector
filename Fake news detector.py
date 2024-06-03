import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load your preprocessed dataset into a pandas DataFrame
# Columns: 'text' (news content), 'label' (0 for real, 1 for fake)

df = pd.read_csv(r'/Users/tanyishen/Training Data.csv')
print(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert the text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize the Passive-Aggressive Classifier
pac_classifier = PassiveAggressiveClassifier(max_iter=25,C=0.1,loss="squared_hinge")
pac_classifier.fit(tfidf_train, y_train)

# Predict on the test set
y_pred = pac_classifier.predict(tfidf_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

#provide stats
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Precision: {(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]))}")
print(f"Recall: {(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0]))}")
print(f"F1 score: {2*(((conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]))*(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0])))/((conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]))+(conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][0]))))}")

#Now, you can enter your own news!
import pickle

# Save the trained Passive-Aggressive Classifier and TF-IDF vectorizer using pickle
with open('pac_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(pac_classifier, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Load the trained model and TF-IDF vectorizer from the files
with open('pac_classifier_model.pkl', 'rb') as model_file:
    loaded_pac_classifier = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_tfidf_vectorizer = pickle.load(vectorizer_file)
your_news=""
while your_news!="stop":
    # Input your own news text
    your_news = input("Enter your news text: ")

    # Preprocess the input news text
    your_news_tfidf = loaded_tfidf_vectorizer.transform([your_news])

    # Predict using the loaded model
    prediction = loaded_pac_classifier.predict(your_news_tfidf)

    if prediction[0] == 0:
    print("The news is classified as real.")
    else:
    print("The news is classified as fake.")
