from flask import Flask, render_template, request, jsonify
import pickle
from flask_cors import CORS
app = Flask(__name__, template_folder="templates")

# Load your trained model and vectorizer
model = pickle.load(open('nb_trained_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Route to serve the HTML file

CORS(app)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data['text']
    
    # Preprocess and predict
    input_vect = vectorizer.transform([user_input])
    prediction = model.predict(input_vect)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

