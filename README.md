# 🎯 Akinator Web Application

An intelligent character guessing web application built using **Python, Flask, and Machine Learning**.

This project asks users a series of yes/no questions and predicts the character they are thinking of based on their answers.

---

## 🚀 Features

- Interactive question-based gameplay
- Machine Learning prediction model
- Session handling using Flask
- Clean and simple user interface
- Real-time character prediction

---

## 🛠 Technologies Used

- Python
- Flask
- HTML & CSS
- Scikit-Learn(tree)
- Pandas & NumPy
- Pickle (for saving ML model)

---

## 📂 Project Structure

## ⚙️ How It Works

1. User starts the game.
2. The system asks multiple yes/no questions.
3. Answers are stored using Flask sessions.
4. After all questions, the ML model predicts the character.
5. The predicted result is displayed.

---

## 🧠 Machine Learning Model

The model is trained on a dataset containing character features.
Each feature represents a specific attribute such as:

- Gender
- Profession
- Nationality
- Status (Alive/Not Alive)
- Category

The model uses classification algorithms to predict the most probable character.

