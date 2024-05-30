import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Dataset
data = pd.read_csv('../messages.csv') # Modify this where you have the data for the ai to train off of

# Split data
messages = data['message']
labels = data['label']

# Transform
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(messages)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class_distribution = y_train.value_counts()
print(class_distribution)

# SMOTE parameters
minority_class_size = class_distribution.min()
smote = SMOTE(sampling_strategy='minority', k_neighbors=min(minority_class_size - 1, 5), random_state=42)

# fix class imbalance
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# train
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, average="weighted")}')
print(f'Recall: {recall_score(y_test, y_pred, average="weighted")}')
print(f'F1-Score: {f1_score(y_test, y_pred, average="weighted")}')


def classify_message(message):
	message_transformed = vectorizer.transform([message])
	prediction = model.predict(message_transformed)
	return prediction[0]


with open("test_messages.txt", "r", encoding="utf-8") as f:
	message_test = f.readlines()
buying_account_int = 0
buying_account = []
asking_for_boost_int = 0
asking_for_boost = []
time_rn = datetime.datetime.now()
for i in message_test:
	thing = classify_message(i)
	if thing != "acceptable":
		print(f'The message "{i.strip()}" is classified as: {thing}')
		if thing == "buying_account":
			buying_account_int += 1
			buying_account.append(i.strip())
		elif thing == "asking_for_boost":
			asking_for_boost_int += 1
			asking_for_boost.append(i.strip())
			
time_end = datetime.datetime.now()


print(f"Total Messages Checked: {len(message_test)}")

print(f"Buying / Selling account: {buying_account_int} ({((buying_account_int / len(message_test)) * 100):.2f}%)")

print(f"Asking for boost: {asking_for_boost_int} ({((asking_for_boost_int / len(message_test)) * 100):.2f}%)")

print(
	f"Total Rule Breaking Messages: {buying_account_int + asking_for_boost_int} / {len(message_test)} "
	f"({(((buying_account_int + asking_for_boost_int) / len(message_test)) * 100):.2f}%)")

time_taken = time_end - time_rn

print(f"\nTotal Time Taken: {time_taken}")

print(f"\nTime Taken Per Message: {(time_taken.total_seconds() / len(message_test)):.4f}")

print(f"Buying / Selling Account Flagged Messages\n------------------")
print(str(buying_account) + "\n------------------\n\n")

print(f"Asking for boost Flagged Messages\n------------------")
print(str(asking_for_boost) + "\n------------------\n\n")
