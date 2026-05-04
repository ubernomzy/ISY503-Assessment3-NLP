from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def predict_sentiment(review):

    """
    Temporary prediction function.

    """

    return "Positive review"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    if review.strip() == "":
        return jsonify({"error": "Please enter a review"}), 400

    result = predict_sentiment(review)

    return jsonify({"sentiment": result})


if __name__ == "__main__":
    app.run(debug=True)