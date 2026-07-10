import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle



#Step 1: Generating the dataset for Disease Information

# =====================================================
# AI-Powered Medical Assistant
# Professional Dataset Generator
# PART 1
# =====================================================

import pandas as pd
import random

random.seed(42)

RECORDS_PER_DISEASE = 40

SYMPTOM_COLUMNS = [
    "Symptom_1",
    "Symptom_2",
    "Symptom_3",
    "Symptom_4",
    "Symptom_5",
    "Symptom_6",
    "Symptom_7",
    "Symptom_8",
    "Symptom_9",
    "Symptom_10",
    "Symptom_11",
    "Symptom_12"
]

records = []

# =====================================================
# Disease Knowledge Base
# =====================================================

disease_data = {

# =====================================================
# 1. Common Cold
# =====================================================

"Common Cold":{

"symptoms":[
"cough",
"runny nose",
"sneezing",
"sore throat",
"nasal congestion",
"mild fever",
"headache",
"fatigue",
"watery eyes",
"hoarse voice",
"ear pressure",
"body ache"
],

"department":"General Medicine",

"severity":"Low",

"risk":"Low",

"advice":"Drink plenty of fluids and take adequate rest.",

"prevention":"Wash hands frequently and avoid close contact with infected people."

},

# =====================================================
# 2. Influenza
# =====================================================

"Influenza":{

"symptoms":[
"high fever",
"cough",
"body pain",
"fatigue",
"headache",
"chills",
"sore throat",
"runny nose",
"muscle pain",
"weakness",
"loss of appetite",
"sweating"
],

"department":"General Medicine",

"severity":"Medium",

"risk":"Medium",

"advice":"Rest, stay hydrated and consult a doctor if symptoms worsen.",

"prevention":"Annual flu vaccination and proper hygiene."

},

# =====================================================
# 3. COVID-19
# =====================================================

"COVID-19":{

"symptoms":[
"fever",
"dry cough",
"loss of smell",
"loss of taste",
"fatigue",
"shortness of breath",
"headache",
"sore throat",
"body pain",
"runny nose",
"diarrhea",
"chest pain"
],

"department":"General Medicine",

"severity":"High",

"risk":"High",

"advice":"Isolate yourself and seek medical advice immediately.",

"prevention":"Vaccination, mask usage and hand hygiene."

},

# =====================================================
# 4. Dengue
# =====================================================

"Dengue":{

"symptoms":[
"high fever",
"headache",
"joint pain",
"muscle pain",
"skin rash",
"vomiting",
"fatigue",
"nausea",
"eye pain",
"loss of appetite",
"bleeding gums",
"weakness"
],

"department":"General Medicine",

"severity":"High",

"risk":"High",

"advice":"Drink fluids and seek immediate medical care.",

"prevention":"Prevent mosquito bites and remove stagnant water."

},

# =====================================================
# 5. Malaria
# =====================================================

"Malaria":{

"symptoms":[
"high fever",
"chills",
"sweating",
"vomiting",
"headache",
"fatigue",
"muscle pain",
"nausea",
"diarrhea",
"abdominal pain",
"loss of appetite",
"weakness"
],

"department":"General Medicine",

"severity":"High",

"risk":"High",

"advice":"Start antimalarial treatment immediately.",

"prevention":"Use mosquito nets and insect repellents."

},
# =====================================================
# 6. Typhoid
# =====================================================

"Typhoid":{

"symptoms":[
"high fever",
"headache",
"abdominal pain",
"vomiting",
"fatigue",
"loss of appetite",
"constipation",
"diarrhea",
"weakness",
"muscle pain",
"chills",
"nausea"
],

"department":"General Medicine",

"severity":"High",

"risk":"High",

"advice":"Take prescribed antibiotics and stay hydrated.",

"prevention":"Drink clean water and maintain food hygiene."

},

# =====================================================
# 7. Pneumonia
# =====================================================

"Pneumonia":{

"symptoms":[
"fever",
"cough",
"chest pain",
"shortness of breath",
"fatigue",
"rapid breathing",
"chills",
"sweating",
"loss of appetite",
"weakness",
"bluish lips",
"confusion"
],

"department":"Pulmonology",

"severity":"High",

"risk":"High",

"advice":"Seek medical attention immediately.",

"prevention":"Vaccination and good respiratory hygiene."

},

# =====================================================
# 8. Tuberculosis
# =====================================================

"Tuberculosis":{

"symptoms":[
"persistent cough",
"weight loss",
"night sweats",
"fever",
"fatigue",
"loss of appetite",
"chest pain",
"coughing blood",
"weakness",
"chills",
"breathing difficulty",
"swollen lymph nodes"
],

"department":"Pulmonology",

"severity":"High",

"risk":"High",

"advice":"Complete the full course of TB medication.",

"prevention":"Early diagnosis and avoid close contact with infected persons."

},

# =====================================================
# 9. Asthma
# =====================================================

"Asthma":{

"symptoms":[
"shortness of breath",
"wheezing",
"cough",
"chest tightness",
"difficulty breathing",
"rapid breathing",
"fatigue",
"anxiety",
"night cough",
"exercise intolerance",
"chest discomfort",
"mild fever"
],

"department":"Pulmonology",

"severity":"Medium",

"risk":"Medium",

"advice":"Use prescribed inhalers and avoid triggers.",

"prevention":"Avoid smoke, dust and allergens."

},

# =====================================================
# 10. Bronchitis
# =====================================================

"Bronchitis":{

"symptoms":[
"persistent cough",
"mucus production",
"fatigue",
"shortness of breath",
"chest discomfort",
"mild fever",
"sore throat",
"body ache",
"wheezing",
"headache",
"runny nose",
"weakness"
],

"department":"Pulmonology",

"severity":"Medium",

"risk":"Medium",

"advice":"Take adequate rest and drink warm fluids.",

"prevention":"Avoid smoking and respiratory infections."

},

# =====================================================
# 11. Diabetes
# =====================================================

"Diabetes":{

"symptoms":[
"frequent urination",
"excessive thirst",
"increased hunger",
"fatigue",
"blurred vision",
"weight loss",
"slow wound healing",
"dry mouth",
"tingling feet",
"tingling hands",
"frequent infections",
"weakness"
],

"department":"Endocrinology",

"severity":"Medium",

"risk":"Medium",

"advice":"Monitor blood sugar regularly and follow prescribed medication.",

"prevention":"Maintain a healthy diet, exercise regularly and control body weight."

},

# =====================================================
# 12. Hypertension
# =====================================================

"Hypertension":{

"symptoms":[
"headache",
"dizziness",
"blurred vision",
"chest pain",
"fatigue",
"nosebleeds",
"shortness of breath",
"palpitations",
"anxiety",
"confusion",
"weakness",
"irregular heartbeat"
],

"department":"Cardiology",

"severity":"Medium",

"risk":"High",

"advice":"Reduce salt intake and take blood pressure medication regularly.",

"prevention":"Exercise regularly, avoid smoking and reduce stress."

},

# =====================================================
# 13. Heart Attack
# =====================================================

"Heart Attack":{

"symptoms":[
"chest pain",
"shortness of breath",
"left arm pain",
"jaw pain",
"sweating",
"nausea",
"vomiting",
"dizziness",
"fatigue",
"palpitations",
"anxiety",
"fainting"
],

"department":"Cardiology",

"severity":"High",

"risk":"High",

"advice":"Seek emergency medical care immediately.",

"prevention":"Maintain healthy lifestyle and manage blood pressure and cholesterol."

},

# =====================================================
# 14. Migraine
# =====================================================

"Migraine":{

"symptoms":[
"severe headache",
"nausea",
"vomiting",
"light sensitivity",
"sound sensitivity",
"blurred vision",
"dizziness",
"neck pain",
"fatigue",
"eye pain",
"irritability",
"difficulty concentrating"
],

"department":"Neurology",

"severity":"Medium",

"risk":"Medium",

"advice":"Rest in a quiet dark room and take prescribed medication.",

"prevention":"Avoid known triggers and maintain regular sleep."

},

# =====================================================
# 15. Kidney Stone
# =====================================================

"Kidney Stone":{

"symptoms":[
"severe back pain",
"flank pain",
"painful urination",
"blood in urine",
"frequent urination",
"nausea",
"vomiting",
"fever",
"chills",
"cloudy urine",
"abdominal pain",
"restlessness"
],

"department":"Urology",

"severity":"High",

"risk":"High",

"advice":"Drink plenty of water and consult a urologist.",

"prevention":"Stay hydrated and reduce excess salt intake."

},
# =====================================================
# 16. Urinary Tract Infection
# =====================================================

"Urinary Tract Infection":{

"symptoms":[
"burning urination",
"frequent urination",
"cloudy urine",
"blood in urine",
"pelvic pain",
"lower abdominal pain",
"fever",
"chills",
"fatigue",
"strong urine smell",
"back pain",
"nausea"
],

"department":"Urology",

"severity":"Medium",

"risk":"Medium",

"advice":"Take antibiotics as prescribed and drink plenty of water.",

"prevention":"Maintain proper hygiene and stay hydrated."

},

# =====================================================
# 17. Gastritis
# =====================================================

"Gastritis":{

"symptoms":[
"stomach pain",
"bloating",
"nausea",
"vomiting",
"loss of appetite",
"indigestion",
"heartburn",
"abdominal discomfort",
"belching",
"burning sensation",
"fatigue",
"hiccups"
],

"department":"Gastroenterology",

"severity":"Medium",

"risk":"Medium",

"advice":"Avoid spicy food and take prescribed medication.",

"prevention":"Eat balanced meals and avoid excessive alcohol."

},

# =====================================================
# 18. Food Poisoning
# =====================================================

"Food Poisoning":{

"symptoms":[
"vomiting",
"diarrhea",
"abdominal pain",
"fever",
"nausea",
"fatigue",
"dehydration",
"stomach cramps",
"weakness",
"headache",
"loss of appetite",
"chills"
],

"department":"Gastroenterology",

"severity":"Medium",

"risk":"Medium",

"advice":"Drink ORS and consult a doctor if dehydration occurs.",

"prevention":"Consume properly cooked and hygienic food."

},

# =====================================================
# 19. Arthritis
# =====================================================

"Arthritis":{

"symptoms":[
"joint pain",
"joint stiffness",
"joint swelling",
"reduced movement",
"redness",
"warm joints",
"fatigue",
"muscle weakness",
"morning stiffness",
"pain while walking",
"back pain",
"difficulty climbing stairs"
],

"department":"Orthopedics",

"severity":"Medium",

"risk":"Medium",

"advice":"Exercise regularly and take prescribed pain medication.",

"prevention":"Maintain healthy weight and stay physically active."

},

# =====================================================
# 20. Anemia
# =====================================================

"Anemia":{

"symptoms":[
"fatigue",
"weakness",
"pale skin",
"shortness of breath",
"dizziness",
"headache",
"cold hands",
"cold feet",
"chest pain",
"rapid heartbeat",
"brittle nails",
"poor concentration"
],

"department":"Hematology",

"severity":"Medium",

"risk":"Medium",

"advice":"Increase iron-rich foods and consult a physician.",

"prevention":"Maintain a balanced diet rich in iron."

}

}

# =====================================================
# DATASET GENERATION
# =====================================================

for disease, info in disease_data.items():

    for _ in range(RECORDS_PER_DISEASE):

        total_symptoms = random.randint(4,7)

        selected = random.sample(info["symptoms"], total_symptoms)

        while len(selected) < 12:
            selected.append("")

        row = {
            "Disease": disease,
            "Department": info["department"],
            "Severity": info["severity"],
            "Risk_Level": info["risk"],
            "Advice": info["advice"],
            "Prevention": info["prevention"]
        }

        for i in range(12):
            row[f"Symptom_{i+1}"] = selected[i]

        records.append(row)


# CREATE DATAFRAME and SAVE TO CSV

df = pd.DataFrame(records)
df.to_csv("medical_dataset.csv", index=False)

print("Medical Dataset Generated Successfully")
print("Total Records :", len(df))
print("Total Diseases :", df["Disease"].nunique())
print(df.head(10))
print(df.shape)
print(df["Disease"].value_counts())
print(df["Department"].value_counts())
print(df["Severity"].value_counts())

#Step2: Data Preprocessing and Cleaning:
#load the dataset
df = pd.read_csv("medical_dataset.csv")
print("Dataset Loaded Successfully!")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns)
print("Data Types:", df.dtypes)
#check for missing values and duplicates
print("Missing Values:", df.isnull().sum())

print("Duplicate Rows:", df.duplicated().sum())

df.to_csv("cleaned_dataset.csv", index=False)

#STEP 3: Basic data Exploration

print("Unique Diseases:", df["Disease"].unique())
print("Unique Departments:", df["Department"].nunique())    
print("Total Records:", len(df))
print(df["Disease"].value_counts())

#Step 4: Exploratory Data Analysis (EDA)
#univariate analysis
#Disease Distribution
plt.figure(figsize=(12,7))
sns.countplot(y="Disease", data=df, order=df["Disease"].value_counts().index)
#This code creates a bar chart that shows the number of patients for each disease, arranged from the most common disease to the least common disease.
plt.title("Disease Distribution")
plt.show()

#Department Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="Department",data=df, order=df["Department"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Department Distribution")
plt.show()

#Severity Distribution
plt.figure(figsize=(8,5))
sns.countplot( x="Severity", data=df, order=df["Severity"].value_counts().index)
plt.title("Severity Distribution")
plt.show()


#STEP 5: Bivariate Analysis
#Department vs Severity Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(pd.crosstab(df["Department"], df["Severity"]), annot=True, cmap="coolwarm")
plt.title("Department vs Severity Heatmap")
plt.show()

#Disease vs Department Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(pd.crosstab(df["Disease"], df["Department"]), annot=True, cmap="Blues")
plt.title("Disease vs Department Heatmap")  
plt.show()

#STEP 6: Machine Learning
#Create Symptom Features
# Symptom Columns
symptom_columns = [
    "Symptom_1",
    "Symptom_2",
    "Symptom_3",
    "Symptom_4",
    "Symptom_5",
    "Symptom_6",
    "Symptom_7",
    "Symptom_8",
    "Symptom_9",
    "Symptom_10",
    "Symptom_11",
    "Symptom_12"
]

# Convert symptoms into list format
symptom_list = []

for _, row in df.iterrows():

    symptoms = []

    for col in symptom_columns:
        symptoms.append(str(row[col]))

    symptom_list.append(symptoms)

print("Sample Symptoms:")
print(symptom_list[:3])

#This code converts each disease record into a list of symptoms, so the Machine Learning model can process them later.

#Convert Symptoms to Numerical Features
mlb = MultiLabelBinarizer()

X = mlb.fit_transform(symptom_list)

print("Feature Matrix Shape:", X.shape)

#Encode Target Variable
le = LabelEncoder()

y = le.fit_transform(df["Disease"])

print("Number of Diseases:", len(le.classes_))

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#Train Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")
#This code converts symptoms and diseases into numbers, splits the dataset into training and testing sets,
#  trains the Random Forest model, and prepares it to predict diseases for new users.

# ==========================================
# Cross Validation
# ==========================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("Average Cross Validation Accuracy:",
      round(cv_scores.mean() * 100, 2), "%")


# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")


#Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("Standard Deviation:",
      round(cv_scores.std() * 100, 2), "%")
#Save Model Files
pickle.dump(
    model,
    open("disease_model.pkl", "wb")
)

pickle.dump(
    mlb,
    open("symptom_encoder.pkl", "wb")
)

pickle.dump(
    le,
    open("label_encoder.pkl", "wb")
)

print("\nModel Saved Successfully!")


