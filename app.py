import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

app = Flask(__name__)
app.secret_key = 'super_secret_banking_key'  # Needed for storing session state for chat

# Load trained ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            credit_score INTEGER,
            geography TEXT,
            gender TEXT,
            age INTEGER,
            tenure INTEGER,
            balance REAL,
            num_products INTEGER,
            has_cr_card INTEGER,
            is_active_member INTEGER,
            estimated_salary REAL,
            prediction TEXT,
            probability REAL,
            ai_summary TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, age, balance, prediction, probability, timestamp FROM predictions ORDER BY id DESC LIMIT 5')
    history = cursor.fetchall()
    conn.close()
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        credit_score = int(request.form['credit_score'])
        geography = request.form['geography']
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_products = int(request.form['num_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])
        estimated_salary = float(request.form['estimated_salary'])

        input_data = pd.DataFrame([[
            credit_score, geography, gender, age, tenure, balance, 
            num_products, has_cr_card, is_active_member, estimated_salary
        ]], columns=[
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ])

        prob = float(model.predict_proba(input_data)[0][1] * 100)
        pred_label = "High Churn Risk" if prob >= 50 else "Retained / Low Risk"

        ai_summary = "AI integration unavailable (API key missing)."
        if client:
            prompt = f"""
            You are an expert Bank Customer Intelligence and Retention Assistant.
            A customer profile was evaluated by our Gradient Boosting churn model.
            - Credit Score: {credit_score}
            - Location: {geography}
            - Gender: {gender}
            - Age: {age}
            - Tenure: {tenure} years
            - Account Balance: ${balance:,.2f}
            - Products Used: {num_products}
            - Has Credit Card: {'Yes' if has_cr_card == 1 else 'No'}
            - Active Member: {'Yes' if is_active_member == 1 else 'No'}
            - Estimated Salary: ${estimated_salary:,.2f}
            - Model Prediction: {pred_label} (Churn Probability: {prob:.2f}%)

            Provide a concise, professional analysis for a relationship manager:
            1. Plain-language reason why this customer is flagged or safe.
            2. Recommended next action (e.g., Hold, Verify, Escalate, Clear / Retention Offer).
            3. Personalized retention offer copy to send to the customer.
            """
            try:
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt
                )
                ai_summary = response.text
            except Exception as e:
                ai_summary = f"Customer Churn Probability: {prob:.2f}%. Recommended Action: Review account activity. (Error: {e})"

        # Save customer context to session for the interactive chat
        session['customer_context'] = {
            'credit_score': credit_score, 'geography': geography, 'gender': gender,
            'age': age, 'tenure': tenure, 'balance': balance, 'num_products': num_products,
            'has_cr_card': has_cr_card, 'is_active_member': is_active_member,
            'estimated_salary': estimated_salary, 'pred_label': pred_label, 'prob': prob
        }
        session['chat_history'] = [{'role': 'assistant', 'text': ai_summary}]

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (credit_score, geography, gender, age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary, prediction, probability, ai_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (credit_score, geography, gender, age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary, pred_label, prob, ai_summary))
        conn.commit()
        conn.close()

        return render_template('result.html', pred=pred_label, prob=round(prob, 2), summary=ai_summary, data=request.form, chat_history=session['chat_history'])

    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form.get('user_message', '')
    if not user_msg:
        return redirect(url_for('index'))

    context = session.get('customer_context', {})
    chat_history = session.get('chat_history', [])

    chat_history.append({'role': 'user', 'text': user_msg})

    if client and context:
        system_instruction = f"""
        You are an expert Bank Customer Intelligence and Retention Chatbot assistant discussing this specific customer:
        - Credit Score: {context.get('credit_score')}
        - Geography: {context.get('geography')}
        - Age: {context.get('age')}
        - Balance: ${context.get('balance'):,.2f}
        - Products: {context.get('num_products')}
        - Status: {context.get('pred_label')} ({context.get('prob'):.2f}% churn probability)
        
        Answer the relationship manager's follow-up question helpfully and professionally.
        """
        
        conversation_prompt = system_instruction + "\n\nUser Question: " + user_msg
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=conversation_prompt
            )
            bot_reply = response.text
        except Exception as e:
            bot_reply = f"Error communicating with Gemini API: {e}"
    else:
        bot_reply = "Chat session context expired or API key missing."

    chat_history.append({'role': 'assistant', 'text': bot_reply})
    session['chat_history'] = chat_history

    return render_template('result.html', 
                           pred=context.get('pred_label'), 
                           prob=round(context.get('prob', 0), 2), 
                           summary=chat_history[0]['text'], 
                           data=context, 
                           chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)