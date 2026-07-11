# ==========================================
# AI-Based Disease Prediction System
# with Medical Chatbot and Healthcare Analytics
# ==========================================

from unittest import result

import streamlit as st
import pandas as pd
import plotly.express as px

from predict_disease import predict_disease
from deep_translator import GoogleTranslator

from symptoms_map import SMART_SYMPTOMS

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="AI-Powered Medical Assistant",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------
# Load Dataset
# ------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("medical_dataset.csv")

df = load_data()

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.title("🏥 AI-Powered Medical Assistant")

# ==========================================
# Language Selection
# ==========================================

LANGUAGES = {
    "English 🇬🇧": "en",
    "Hindi 🇮🇳": "hi"
    }
HINDI_DEPARTMENT = {
    "General Medicine": "सामान्य चिकित्सा",
    "Pulmonology": "श्वसन रोग विभाग",
    "Endocrinology": "अंतःस्रावी रोग विभाग",
    "Cardiology": "हृदय रोग विभाग",
    "Neurology": "तंत्रिका रोग विभाग",
    "Urology": "मूत्र रोग विभाग",
    "Gastroenterology": "गैस्ट्रोएंटरोलॉजी विभाग",
    "Orthopedics": "अस्थि रोग विभाग",
    "Hematology": "रक्त रोग विभाग"
}
HINDI_ADVICE = {

"Drink plenty of fluids and take adequate rest.":
"पर्याप्त मात्रा में तरल पदार्थ पिएं और पर्याप्त आराम करें।",

"Rest, stay hydrated and consult a doctor if symptoms worsen.":
"आराम करें, पर्याप्त पानी पिएं और यदि लक्षण बढ़ जाएं तो डॉक्टर से परामर्श करें।",

"Isolate yourself and seek medical advice immediately.":
"स्वयं को अलग रखें और तुरंत डॉक्टर से सलाह लें।",

"Drink fluids and seek immediate medical care.":
"पर्याप्त तरल पदार्थ पिएं और तुरंत चिकित्सकीय सहायता लें।",

"Start antimalarial treatment immediately.":
"तुरंत मलेरिया का उपचार शुरू करें।",

"Take prescribed antibiotics and stay hydrated.":
"डॉक्टर द्वारा दी गई एंटीबायोटिक दवाएं लें और पर्याप्त पानी पिएं।",

"Seek medical attention immediately.":
"तुरंत चिकित्सकीय सहायता प्राप्त करें।",

"Complete the full course of TB medication.":
"टीबी की दवाओं का पूरा कोर्स पूरा करें।",

"Use prescribed inhalers and avoid triggers.":
"डॉक्टर द्वारा बताए गए इनहेलर का उपयोग करें और ट्रिगर्स से बचें।",

"Take adequate rest and drink warm fluids.":
"पर्याप्त आराम करें और गर्म तरल पदार्थ पिएं।",

"Monitor blood sugar regularly and follow prescribed medication.":
"नियमित रूप से रक्त शर्करा की जांच करें और डॉक्टर की दवाएं लें।",

"Reduce salt intake and take blood pressure medication regularly.":
"नमक कम खाएं और रक्तचाप की दवाएं नियमित रूप से लें।",

"Seek emergency medical care immediately.":
"तुरंत आपातकालीन चिकित्सकीय सहायता प्राप्त करें।",

"Rest in a quiet dark room and take prescribed medication.":
"शांत और अंधेरे कमरे में आराम करें तथा डॉक्टर की दवा लें।",

"Drink plenty of water and consult a urologist.":
"पर्याप्त पानी पिएं और मूत्र रोग विशेषज्ञ से परामर्श करें।",

"Take antibiotics as prescribed and drink plenty of water.":
"डॉक्टर द्वारा बताई गई एंटीबायोटिक दवाएं लें और पर्याप्त पानी पिएं।",

"Avoid spicy food and take prescribed medication.":
"मसालेदार भोजन से बचें और डॉक्टर की दवा लें।",

"Drink ORS and consult a doctor if dehydration occurs.":
"ओआरएस पिएं और यदि शरीर में पानी की कमी हो तो डॉक्टर से परामर्श करें।",

"Exercise regularly and take prescribed pain medication.":
"नियमित व्यायाम करें और डॉक्टर द्वारा बताई गई दर्द की दवा लें।",

"Increase iron-rich foods and consult a physician.":
"आयरन युक्त भोजन अधिक लें और चिकित्सक से परामर्श करें।"

}

HINDI_PREVENTION = {

"Wash hands frequently and avoid close contact with infected people.":
"बार-बार हाथ धोएं और संक्रमित लोगों के निकट संपर्क से बचें।",

"Annual flu vaccination and proper hygiene.":
"हर वर्ष फ्लू का टीका लगवाएं और व्यक्तिगत स्वच्छता बनाए रखें।",

"Vaccination, mask usage and hand hygiene.":
"टीकाकरण कराएं, मास्क पहनें और हाथों की स्वच्छता बनाए रखें।",

"Prevent mosquito bites and remove stagnant water.":
"मच्छरों के काटने से बचें और रुके हुए पानी को हटाएं।",

"Use mosquito nets and insect repellents.":
"मच्छरदानी और मच्छर भगाने वाले रिपेलेंट का उपयोग करें।",

"Drink clean water and maintain food hygiene.":
"स्वच्छ पानी पिएं और भोजन की स्वच्छता बनाए रखें।",

"Vaccination and good respiratory hygiene.":
"टीकाकरण कराएं और श्वसन स्वच्छता बनाए रखें।",

"Early diagnosis and avoid close contact with infected persons.":
"समय पर जांच कराएं और संक्रमित व्यक्तियों के निकट संपर्क से बचें।",

"Avoid smoke, dust and allergens.":
"धुएं, धूल और एलर्जी पैदा करने वाले पदार्थों से बचें।",

"Avoid smoking and respiratory infections.":
"धूम्रपान और श्वसन संक्रमण से बचें।",

"Maintain a healthy diet, exercise regularly and control body weight.":
"संतुलित आहार लें, नियमित व्यायाम करें और वजन नियंत्रित रखें।",

"Exercise regularly, avoid smoking and reduce stress.":
"नियमित व्यायाम करें, धूम्रपान से बचें और तनाव कम रखें।",

"Maintain healthy lifestyle and manage blood pressure and cholesterol.":
"स्वस्थ जीवनशैली अपनाएं और रक्तचाप तथा कोलेस्ट्रॉल नियंत्रित रखें।",

"Avoid known triggers and maintain regular sleep.":
"ज्ञात ट्रिगर्स से बचें और नियमित नींद लें।",

"Stay hydrated and reduce excess salt intake.":
"पर्याप्त पानी पिएं और अधिक नमक खाने से बचें।",

"Maintain proper hygiene and stay hydrated.":
"स्वच्छता बनाए रखें और पर्याप्त पानी पिएं।",

"Eat balanced meals and avoid excessive alcohol.":
"संतुलित भोजन करें और अत्यधिक शराब के सेवन से बचें।",

"Consume properly cooked and hygienic food.":
"अच्छी तरह पका हुआ और स्वच्छ भोजन करें।",

"Maintain healthy weight and stay physically active.":
"स्वस्थ वजन बनाए रखें और शारीरिक रूप से सक्रिय रहें।",

"Maintain a balanced diet rich in iron.":
"आयरन युक्त संतुलित आहार लें।"

}



selected_language = st.sidebar.selectbox(
    "🌍 Choose Your Preferred Language",
    list(LANGUAGES.keys())
)

st.sidebar.success(
    f"🌍 Current Language : {selected_language}"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Healthcare Analytics",
        "🤖 Medical Chatbot",
        "ℹ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
💡 Quick Tip

Enter symptoms separated by commas.

Example:

fever, cough, headache
"""
)

# ==========================================
# HOME PAGE
# ==========================================
def home_page():

# ==========================================
# TITLE
# ==========================================

    st.title("🏥 AI-Powered Medical Assistant")
    st.subheader("**For Disease Prediction and Healthcare Analytics**")
    
    st.markdown("---")

    # ==========================================
    # BANNER IMAGE
    # ==========================================

    col1, col2, col3 = st.columns([1,6,1])

    with col2:

        st.image(
            "Medbot3.png",   
            width=950
        )
    
    st.markdown("---")

    # ==========================================
    # WELCOME
    # ==========================================

    st.header("👋 Welcome to MediBot")

    st.write("""
MediBot is an AI-powered healthcare assistant that predicts diseases
based on the symptoms entered by the user.

The system uses Machine Learning to provide quick and reliable
healthcare guidance.
""")

    st.markdown("---")
  

    # ==========================================
    # FEATURES
    # ==========================================

    st.subheader("✨ Project Features")

    c1, c2 = st.columns(2)

    with c1:

        st.success("🦠 Disease Prediction")

        st.success("🤖 Medical Chatbot")

        st.success("🏥 Department Recommendation")

        st.success("💊 Medical Advice")

    with c2:

        st.success("🛡 Prevention Tips")

        st.success("📊 Healthcare Analytics")

        st.success("🤖 Bilingual Support (English & Hindi)")

        st.success("🔒 Secure & User Friendly")


    st.markdown("---")

    st.info(" ✨ Please explore the application using the navigation menu on the left to access different features.")
    # ==========================================
    # TECHNOLOGY
    # ==========================================

    st.subheader("💻 Technologies Used")

    t1, t2 = st.columns(2)

    with t1:

        st.info("""
🐍 Python

🔢 NumPy

🐼 Pandas

🤖 Scikit-Learn
""")

    with t2:

        st.info("""
🎈  Streamlit

🌐 Deep Translator

📈 Plotly

📉 Matplotlib
""")

    st.markdown("---")

    # ==========================================
    # FOOTER
    # ==========================================

    st.success("""
❤️ Thank you for using MediBot.

This project is developed for educational purposes only.

Always consult a qualified healthcare professional for medical advice.
""")
# ==========================================
# DISEASE PREDICTION PAGE
# ==========================================

def disease_prediction_page():

    st.title("🔍 Disease Prediction")

    st.write(
        "Enter symptoms separated by commas."
    )

    symptoms = st.text_area(
        "Symptoms",
        placeholder="Example: fever, headache, cough"
    )

    if st.button("🔍 Predict Disease", width='stretch'):

        if symptoms.strip() == "":
            if selected_language == "English 🇬🇧":
                st.warning("Please enter at least 4 symptoms for accurate prediction.")

            st.warning("Please enter at least one symptom.")

        else:

            symptom_list = [
                s.strip().lower()
                for s in symptoms.split(",")
                if s.strip()
            ]

            if len(symptom_list) < 4:
                if selected_language == "English 🇬🇧":
                    st.warning("Please enter at least 4 symptoms for accurate prediction.")
                else:
                    st.warning("कृपया सटीक भविष्यवाणी के लिए कम से कम 4 लक्षण दर्ज करें।") 
            else:       
              result = predict_disease(symptom_list)

            st.divider()

            col1, col2 = st.columns(2)

            # -------------------------
            # Disease
            # -------------------------

            with col1:

                st.success(
                    f"### 🦠 Disease\n\n{result['Disease']}"
                )

            # -------------------------
            # Department
            # -------------------------

            with col2:

                st.info(
                    f"### 🏥 Department\n\n{result['Department']}"
                )

            # -------------------------
            # Severity
            # -------------------------

            severity = result["Severity"].lower()

            if severity == "low":

                st.success(f"### 🟢 Severity\n\n{result['Severity']}")

            elif severity == "medium":

                st.warning(f"### 🟡 Severity\n\n{result['Severity']}")

            else:

                st.error(f"### 🔴 Severity\n\n{result['Severity']}")

            # -------------------------
            # Advice
            # -------------------------

            st.subheader("💊 Medical Advice")

            st.info(result["Advice"])

            # -------------------------
            # Prevention
            # -------------------------

            st.subheader("🛡 Prevention")

            st.success(result["Prevention"])

            st.divider()

            st.caption(
                "⚠ This prediction is intended for educational purposes only and should not replace professional medical advice."
            )

# ==========================================
# HEALTHCARE ANALYTICS PAGE
# ==========================================

def analytics_page():

    st.title("📊 Healthcare Analytics Dashboard")

    st.write(
        "Explore disease patterns and healthcare insights from the medical dataset."
    )

    st.divider()

    # ----------------------------
    # Disease Distribution
    # ----------------------------

    st.subheader("🦠 Disease Distribution")

    disease_count = (
        df["Disease"]
        .value_counts()
        .reset_index()
    )

    disease_count.columns = ["Disease", "Count"]

    fig = px.bar(
        disease_count,
        x="Disease",
        y="Count",
        color="Count",
        title="Disease Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Department Distribution
    # ----------------------------

    st.subheader("🏥 Department Distribution")

    dept_count = (
        df["Department"]
        .value_counts()
        .reset_index()
    )

    dept_count.columns = ["Department", "Count"]

    fig = px.pie(
        dept_count,
        names="Department",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------
    # Severity Distribution
    # ----------------------------

    st.subheader("🚨 Severity Distribution")

    severity_count = (
        df["Severity"]
        .value_counts()
        .reset_index()
    )

    severity_count.columns = ["Severity", "Count"]

    fig = px.bar(
        severity_count,
        x="Severity",
        y="Count",
        color="Severity",
        title="Disease Severity"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MEDIBOT
# ==========================================

def chatbot_page():

    # ----------------------------
    # Bot Image
    # ----------------------------

    st.image(
        "Medibot2.png",
        width=650
    )
    st.subheader("Disease Prediction • Healthcare Analytics • Medical Chatbot")
    # ----------------------------
    # Welcome Message
    # ----------------------------

    st.info("""
👋 Hi!

I'm **MediBot**, your AI Healthcare Assistant.

I can help predict possible diseases based on the symptoms you enter.

Simply enter your symptoms below and I'll provide:

✅ Predicted Disease

✅ Recommended Department

✅ Medical Advice

✅ Prevention Tips

Please enter your symptoms below to get started.
""")

    st.divider()

    # ----------------------------
    # User Input
    # ----------------------------

    if selected_language == "English 🇬🇧":
     user_input = st.text_input(
       "📝 Please Enter at least 3-4 symptoms",
        placeholder="Example: fever, cough, headache,nausea"
    )
    else:
     user_input = st.text_input(
       "📝 कृपया कम से कम 3-4 लक्षण दर्ज करें",
        placeholder="उदाहरण: बुखार, खांसी, सिरदर्द,मतली"
    )

    # ==========================================
    # Translate User Input
    # ==========================================

    translated_input = user_input

    if user_input.strip():

        if LANGUAGES[selected_language] != "en":

            try:

                translated_input = GoogleTranslator(
                    source="auto",
                    target="en"
                ).translate(user_input)

            except Exception:

                translated_input = user_input

    # ----------------------------
    # Analyze Button
    # ----------------------------

    if st.button(
        "💬 Analyze Symptoms",
         width='stretch'
    ):

        message = translated_input.strip().lower()

        # ==========================================
        # Normalize Symptoms using Symptom Map
        # ==========================================

        expanded_symptoms = []

        for symptom in message.split(","):

            symptom = symptom.strip().lower()

            if symptom in SMART_SYMPTOMS:

                expanded_symptoms.extend(
                    SMART_SYMPTOMS[symptom]
                )

            else:

                expanded_symptoms.append(symptom)

        message = ",".join(expanded_symptoms)

        # ----------------------------
        # Empty Input
        # ----------------------------

        if message == "":

            st.warning(
                "Please enter your symptoms so I can assist you."
            )

            return

        # ----------------------------
        # Thanks
        # ----------------------------

        if message in [
            "thanks",
            "thank you",
            "thankyou"
        ]:

            st.success("""
🤖 MediBot

You're welcome 😊

Take care and stay healthy.
""")

            return

        # ----------------------------
        # Goodbye
        # ----------------------------

        if message in [
            "bye",
            "goodbye",
            "see you"
        ]:

            st.success("""
🤖 MediBot

Thank you for using MediBot.

Have a wonderful day.

Stay healthy 🌿
""")

            return

        # ----------------------------
        # Show Translation
        # ----------------------------

        if translated_input.lower() != user_input.lower():

            st.success("🧠 Symptoms Identified")

            st.info(translated_input)

        # ----------------------------
        # Symptoms List
        # ----------------------------

        symptoms = [

            s.strip()

            for s in message.split(",")

            if s.strip()

        ]

        # ----------------------------
        # Prediction Spinner
        # ----------------------------

        with st.spinner(
            "🤖 MediBot is analyzing your symptoms..."
        ):

            result = predict_disease(symptoms)
        disease = result["Disease"]
        department = result["Department"]
        severity = result["Severity"]
        advice = result["Advice"]
        prevention = result["Prevention"]    

        st.divider()

        # ==========================================
        # English Response
        # ==========================================

        if LANGUAGES[selected_language] == "en":

            st.success(f"""
🤖 **MediBot**

Based on the symptoms you entered, the most likely disease is:

❤️ **{disease}**

This is an AI-based preliminary prediction.

Please consult a qualified doctor for an accurate diagnosis.
""")

            st.subheader("🩺 Predicted Disease")
            st.success(disease)

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"🏥 **Department**\n\n{department}")
                
            with col2:

                if severity.lower() == "low":
                    st.success(f"🟢 **Severity**\n\n{severity}")

                elif severity.lower() == "medium":
                    st.warning(f"🟡 **Severity**\n\n{severity}")

                else:
                    st.error(f"🔴 **Severity**\n\n{severity}")

            st.subheader("💊 Medical Advice")
            st.success(advice)
            st.subheader("🛡 Prevention Tips")
            st.info(prevention)

            st.warning("""
⚠ **Medical Disclaimer**

This prediction is generated using Machine Learning.

Please consult a qualified healthcare professional.
""")

            st.success("""
❤️ Thank you for trusting MediBot.

Stay Healthy 🌿
""")

        # ==========================================
        # Hindi Response
        # ==========================================

        else:

            st.success(f"""
🤖 **MediBot**

आपके द्वारा बताए गए लक्षणों के आधार पर संभावित बीमारी है:

❤️ **{disease}**

यह केवल AI आधारित प्रारंभिक अनुमान है।

सही निदान के लिए कृपया योग्य डॉक्टर से परामर्श करें।
""")

            st.subheader("🩺 संभावित बीमारी")
            st.info(HINDI_DEPARTMENT.get(department, department))

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"🏥 **संबंधित विभाग**\n\n{department}")

            with col2:

                if severity.lower() == "low":
                    st.success(f"🟢 **गंभीरता**\n\n{severity}")

                elif severity.lower() == "medium":
                    st.warning(f"🟡 **गंभीरता**\n\n{severity}")

                else:
                    st.error(f"🔴 **गंभीरता**\n\n{severity}")

            st.subheader("💊 चिकित्सकीय सलाह")
            st.success(HINDI_ADVICE.get(advice, advice))

            st.subheader("🛡 बचाव के उपाय")
            st.info(HINDI_PREVENTION.get(prevention, prevention))

            st.warning("""
⚠ **चिकित्सकीय अस्वीकरण**

यह केवल AI आधारित प्रारंभिक अनुमान है।

सही निदान के लिए कृपया योग्य डॉक्टर से परामर्श करें।
""")

            st.success("""
❤️ MediBot पर विश्वास करने के लिए धन्यवाद।

स्वस्थ रहें 🌿
""")
 # ==========================================
# ABOUT PAGE
# ==========================================

def about_page():

    st.title("ℹ About Project")

    st.markdown("""
## 🏥 AI-Based Disease Prediction System
### with Medical Chatbot and Healthcare Analytics

This application is a Machine Learning-based healthcare decision support system
developed to predict diseases from symptoms entered by the user.

### ✨ Project Modules

- 🔍 Disease Prediction
- 🤖 AI Medical Chatbot (MediBot)
- 📊 Healthcare Analytics
- 💊 Medical Advice
- 🛡 Prevention Tips

### 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Plotly
- Machine Learning

### 👨‍💻 Purpose

The objective of this project is to demonstrate how Machine Learning can assist
users by providing preliminary healthcare guidance based on symptoms.

This application is intended for educational purposes only and should not be
used as a substitute for professional medical consultation.
""")

    st.divider()

    st.success("""
Thank you for using the AI-Based Disease Prediction System.

Stay Healthy ❤️
""") 


 # ==========================================
# PAGE NAVIGATION
# ==========================================

if page == "🏠 Home":
    home_page()

elif page == "🔍 Disease Prediction":
    disease_prediction_page()

elif page == "📊 Healthcare Analytics":
    analytics_page()

elif page == "🤖 Medical Chatbot":
    chatbot_page()

elif page == "ℹ About Project":
    about_page()             

 
