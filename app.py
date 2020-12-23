from flask import Flask, render_template, request
import SentimentAnalysis.sentiment_model as sentiment_model

app = Flask(__name__)

global sentiment
global tokenizer
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    # return str('Sentiment is - {}'.format(sentiment_predictions([userText])))
    # return str(sentiment_predictions([userText]))
    global sentiment
    return sentiment.predictions([userText], tokenizer)

def main():
    global sentiment
    global tokenizer
    sentiment = sentiment_model.SentimentModel()
    tokenizer = sentiment.main()

if __name__ == "__main__":
    main()
    app.run()