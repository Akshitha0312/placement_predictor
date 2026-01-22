from flask import Flask, render_template, request
import pickle
import os
from flask import send_file
app = Flask(__name__)

@app.route("/download")
def download():
    return send_file("history.csv", as_attachment=True)


model = pickle.load(open("model.pkl", "rb"))
# print(model)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    coding = int(request.form["coding"])
    communication = int(request.form["communication"])

    result = model.predict([[cgpa, internships, coding, communication]])

    # Probability %
    prob = model.predict_proba([[cgpa, internships, coding, communication]])[0][1]
    prob = round(prob * 100, 2)
    
    if result[0] == 1:
        msg = "High chance of placement 🎉"
    else:
        msg = "Low chance. Improve skills 💪"

    # Recommendations
    tips = []
    if cgpa < 7:
        tips.append("Try to improve CGPA")
    if internships == 0:
        tips.append("Do at least one internship")
    if coding < 3:
        tips.append("Improve coding skills (DSA, Python, Java)")
    if communication < 3:
        tips.append("Work on communication skills")

    # 🔥 THIS WAS MISSING
    with open("history.csv", "a") as f:
        f.write(f"{cgpa},{internships},{coding},{communication},{prob}\n")

    return render_template(
        "result.html",
        message=msg,
        probability=prob,
        cgpa=cgpa,
        internships=internships,
        coding=coding,
        communication=communication,
        tips=tips
    )

if __name__ == "__main__":
    app.run(debug=True,port=5001)
