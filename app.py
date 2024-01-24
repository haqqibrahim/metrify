from flask import Flask, jsonify, request
from model_loader import load_model
from data_preprocessing import preprocess_data
from prediction import make_predictions
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = 'http://petstore.swagger.io/v2/swagger.json'  # Our API url (can of course be a local resource)

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Metrify Endpoint"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load the pre-trained model
model = load_model()

@app.route('/')
def hello():
    return 'Hello, World, Welcome to Metrify!!!!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the input data is sent as JSON in the request body
        data = request.json

        # Preprocess the input data
        X = preprocess_data(data)

        # Make predictions
        predictions = make_predictions(model, X)

        # Convert predictions to a JSON response
        response = {
            'predictions': predictions.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
