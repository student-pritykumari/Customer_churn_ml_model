import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load trained model safely
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    messagebox.showerror("Error", "Model file not found! Train model first.")
    exit()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_churn():
    try:
        age = int(entry_age.get())
        gender = 1 if gender_var.get() == "Male" else 0
        monthly = float(entry_monthly.get())
        contract = 0 if contract_var.get() == "Monthly" else 1
        
        usage_map = {"Low": 0, "Medium": 1, "High": 2}
        usage = usage_map[usage_var.get()]
        
        calls = int(entry_calls.get())
        satisfaction = int(entry_satisfaction.get())
        
        payment_map = {"UPI": 0, "Card": 1, "Cash": 2}
        payment = payment_map[payment_var.get()]
        
        service_map = {"Streaming": 0, "Telecom": 1, "Banking": 2, "E-commerce": 3}
        service = service_map[service_var.get()]
        
        tenure = int(entry_tenure.get())

        # Input array (IMPORTANT: order must match training)
        data = np.array([[age, gender, monthly, contract,
                          usage, calls, satisfaction,
                          payment, tenure, service]])

        result = model.predict(data)
        prob = model.predict_proba(data)

        if result[0] == 1:
            result_label.config(
                text=f"⚠️ Customer will CHURN\nProbability: {prob[0][1]:.2f}",
                fg="red"
            )
        else:
            result_label.config(
                text=f"✅ Customer will STAY\nProbability: {prob[0][1]:.2f}",
                fg="green"
            )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# -----------------------------
# UI Design
# -----------------------------
root = tk.Tk()
root.title("Customer Churn Prediction System")
root.geometry("420x650")
root.config(bg="#f0f0f0")

title = tk.Label(root, text="Customer Churn Prediction", 
                 font=("Arial", 16, "bold"), bg="#f0f0f0")
title.pack(pady=10)

# Helper function for labels
def add_label(text):
    tk.Label(root, text=text, bg="#f0f0f0").pack()

# Age
add_label("Age")
entry_age = tk.Entry(root)
entry_age.pack()

# Gender
add_label("Gender")
gender_var = tk.StringVar(value="Male")
tk.OptionMenu(root, gender_var, "Male", "Female").pack()

# Monthly Charges
add_label("Monthly Charges")
entry_monthly = tk.Entry(root)
entry_monthly.pack()

# Contract Type
add_label("Contract Type")
contract_var = tk.StringVar(value="Monthly")
tk.OptionMenu(root, contract_var, "Monthly", "Yearly").pack()

# Internet Usage
add_label("Internet Usage")
usage_var = tk.StringVar(value="Low")
tk.OptionMenu(root, usage_var, "Low", "Medium", "High").pack()

# Support Calls
add_label("Support Calls")
entry_calls = tk.Entry(root)
entry_calls.pack()

# Satisfaction
add_label("Satisfaction (1-5)")
entry_satisfaction = tk.Entry(root)
entry_satisfaction.pack()

# Payment Method
add_label("Payment Method")
payment_var = tk.StringVar(value="UPI")
tk.OptionMenu(root, payment_var, "UPI", "Card", "Cash").pack()

# Tenure
add_label("Tenure (months)")
entry_tenure = tk.Entry(root)
entry_tenure.pack()

# NEW: Service Type ⭐
add_label("Service Type")
service_var = tk.StringVar(value="Streaming")
tk.OptionMenu(root, service_var, "Streaming", "Telecom", "Banking", "E-commerce").pack()

# Predict Button
tk.Button(root, text="Predict", command=predict_churn,
          bg="blue", fg="white", width=15).pack(pady=15)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f0f0f0")
result_label.pack(pady=10)

# Run App
root.mainloop()