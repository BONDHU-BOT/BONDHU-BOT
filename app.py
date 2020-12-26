from flask import Flask, render_template, request
import SentimentAnalysis.sentiment_model as sentiment_model
import IntentClassification.intent_model as intent_model
import NamedEntityRecognition.ner_model as ner_model

app = Flask(__name__)

global sentiment
global sentiment_tokenizer
global intent
global intent_tokenizer
global model_intent
global unique_intent
global named_entity

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    global sentiment
    predicted_sentiment = sentiment.predictions([userText], sentiment_tokenizer)

    global intent
    global model_intent
    global unique_intent
    predicted_intent = intent.get_final_prediction(userText, intent_tokenizer, model_intent, unique_intent)
    
    global named_entity
    predicted_ner = named_entity.get_prediction(userText)
    return str("predicted sentiment - " + predicted_sentiment + ".\n\npredicted intent - " + predicted_intent + ".\n\npredicted named-entity - " + predicted_ner)

def get_sentiment():
    global sentiment
    global sentiment_tokenizer
    sentiment = sentiment_model.SentimentModel()
    sentiment_tokenizer = sentiment.main()

def get_intent():
    global intent
    global intent_tokenizer
    global model_intent
    global unique_intent
    intent = intent_model.IntentModel()
    intent_tokenizer, model_intent, unique_intent = intent.main()

def get_ner():
    global named_entity
    named_entity = ner_model.NERModel()
    named_entity.main()

def main():
    print('\nplease wait ...\n')
    get_sentiment()
    print('\nplease wait ...\n')
    get_intent()
    print('\nplease wait ...\n')
    get_ner()

if __name__ == "__main__":
    main()
    app.run(debug=True)