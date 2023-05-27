from flask import Flask, render_template, request, jsonify
from model import model_predict,preprocess_text,load_model


app = Flask(__name__)


model = None

def load_trained_model():
    global model
    model = load_model()

load_trained_model()

@app.before_request
def before_request():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if content_type and 'application/json' not in content_type and 'multipart/form-data' not in content_type:
            return jsonify({'error': 'Unsupported Media Type'}), 415
        

@app.route('/', methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            text = data.get('text')
        else:
            text = request.form['text']
        
        # preprocess the input text
        processed_text = preprocess_text(text)

        # make a prediction using the loaded model
        response = model_predict(processed_text,model)
       
        return render_template('index.html', text=text, sentiment=response['sentiment'],scores=response['score'])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
