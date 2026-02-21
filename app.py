from flask import Flask, render_template, request, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load trained Decision Tree model
model = pickle.load(open("akinator_model.pkl", "rb"))

questions = [
    "Is the character real?",
    "Is the character female?",
    "Is the character famous?",
    "Is the character from a movie?",
    "Is the character from a cartoon?",
    "Is the character older than 30?",
    "Is the character a superhero?",
    "Is the character evil?",
    "Is the character rich?",
    "Is the character from Asia?",
    "Is the character a singer?",
    "Is the character an actor?",
    "Is the character fictional?",
    "Is the character from a game?",
    "Is the character married?",
    "Is the character powerful?",
    "Is the character funny?",
    "Is the character human?",
    "Is the character alive?"
]

@app.route("/")
def home():
    session["answers"] = []
    session["index"] = 0
    return render_template("index.html",
                           question=questions[0],
                           progress=1,
                           total=len(questions))

@app.route("/answer", methods=["POST"])
def answer():
    answer = request.form.get("answer")

    # Convert yes/no to 1/0
    numeric = 1 if answer == "yes" else 0
    session["answers"].append(numeric)
    session["index"] += 1

    # If finished 19 questions
    if session["index"] >= len(questions):

        final_input = np.array([session["answers"]])
        prediction = model.predict(final_input)

        character_name = prediction[0]

        return render_template("index.html", result=character_name)

    return render_template("index.html",
                           question=questions[session["index"]],
                           progress=session["index"] + 1,
                           total=len(questions))

if __name__ == "__main__":
    app.run(debug=True)