from flask import Flask, render_template, request
import pickle
import json
import random
# from Mod_chatbot import 

app = Flask(__name__)
words = pickle.load(open('C:/Users/rockstar/OneDrive/Documents/Sholabs/words.pkl', 'rb'))

# Load the trained model and vectorizer
best_model = pickle.load(open('C:/Users/rockstar/OneDrive/Documents/Sholabs/prj1/JarvisChatbot/model/chatbot_model.pkl', 'rb'))

vectorizer = pickle.load(open('C:/Users/rockstar/OneDrive/Documents/Sholabs/prj1/JarvisChatbot/model/vectorizer.pkl', 'rb'))

# Load the intents data
intents = json.load(open('C:/Users/rockstar/OneDrive/Documents/Sholabs/prj1/JarvisChatbot/dataset/intents1.json', 'r'))
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('data.csv')
# Assuming the target variable is 'fail' and features include the sensor readings and other metrics
X = df.drop('fail', axis=1)  # Feature variables
y = df['fail']  # Target variable (machine failure)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (using GradientBoostingClassifier as an example)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")
def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    print(input_text)
    predicted_intent = best_model.predict(input_text)[0]
    print(predicted_intent)
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            msg=str(response)
            break
    if predicted_intent == "Total_Failures":
        response=df['fail'].sum()
        msg="Total machines about to Fail is: "+str(response)
    elif predicted_intent == "Total_Non_Failures":
        response=(df['fail'] == 0).sum()
        msg="Total working machines are: "+str(response)
    elif predicted_intent == "Common_Failure_Conditions":
        response=(df[df['fail'] == 1].mode().iloc[0]).to_dict()
        msg="Common Failure Conditions are: "+str(response)
    elif predicted_intent == "Average_Temperature":
        response=df[df['fail'] == 1]['Temperature'].mean()
        msg="Average Temperature at Failure:"+str(response)
    elif predicted_intent == "High_VOC":
        high_voc_failures = df[(df['VOC'] > 50) & (df['fail'] == 1)]
        total_high_voc = df[df['VOC'] > 50]
        if len(total_high_voc) == 0:
            response= 0
        response= len(high_voc_failures) / len(total_high_voc) * 100
        msg="Failure Rate with High VOC:"+str(response)
    elif predicted_intent == "High_Footfall":
        high_footfall_failures = df[(df['footfall'] > 200) & (df['fail'] == 1)]
        total_high_footfall = df[df['footfall'] > 200]
        if len(total_high_footfall) == 0:
            response= 0
        response= len(high_footfall_failures) / len(total_high_footfall) * 100
        msg="Failure Rate by High Footfall: "+ str(response)
    elif predicted_intent == "CS_Level":
        response=len(df[(df['CS'] > 5) & (df['fail'] == 1)])
        msg="Failures with CS Level: "+ str(response)
    elif predicted_intent == "Poor_Air_Quality":
        poor_aq_failures = df[(df['AQ'] > 70) & (df['fail'] == 1)]
        total_poor_aq = df[df['AQ'] > 70]
        if len(total_poor_aq) == 0:
            response= 0
        response= len(poor_aq_failures) / len(total_poor_aq) * 100
        msg="Failure Rate with Poor Air Quality: "+str(response)
    elif predicted_intent == "High_IP":
        high_ip_failures = df[(df['IP'] > 500) & (df['fail'] == 1)]
        total_high_ip = df[df['IP'] > 500]
        if len(total_high_ip) == 0:
            response= 0
        response= len(high_ip_failures) / len(total_high_ip) * 100
        msg="Failure Rate by High IP: "+str(response)


    
    return msg

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)