import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

class IntentModel:

    max_len = None

    def load_dataset(self, filename, Sentence, Intent):
        df = pd.read_csv(filename, names = [Sentence, Intent])
        intent = df[Intent]
        unique_intent = list(set(intent))
        sentences = list(df[Sentence])
        return (df, intent, unique_intent, sentences)

    def cleaning(self, sentences):
        words = []
        for s in sentences:
            clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
            w = word_tokenize(clean)
            words.append([i.lower() for i in w])
        return words  

    def create_tokenizer(self, words, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'):
        token = Tokenizer(filters = filters)
        token.fit_on_texts(words)
        return token
    
    def max_length(self, words):
        return(len(max(words, key = len)))

    def encoding_doc(self, token, words):
        return(token.texts_to_sequences(words))

    def padding_doc(self, encoded_doc, max_length):
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

    def main(self):
        _, _, unique_intent, sentences = self.load_dataset("IntentClassification/data/Dataset.csv", "text", "category")
        
        cleaned_words = self.cleaning(sentences)
        word_tokenizer = self.create_tokenizer(cleaned_words)
        # vocab_size = len(word_tokenizer.word_index) + 1
        global max_len
        max_len = self.max_length(cleaned_words)

        model = load_model("IntentClassification/model.h5")

        return word_tokenizer, model, unique_intent

    
    ##### PREDICTIONS !!!!!!!!!!!!!!

    def predictions(self, text, word_tokenizer, model_intent):

        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = word_tokenizer.texts_to_sequences(test_word)
        #Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))
        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        x = self.padding_doc(test_ls, max_len)
        pred = model_intent.predict(x)
        return pred

    def get_final_output(self, pred, classes):
        predictions = pred[0]
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)
        # for i in range(pred.shape[1]):
        #     print("%s has confidence = %s" % (classes[i], (predictions[i])))
        return classes[0]

    def get_final_prediction(self, text, word_tokenizer, model_intent, unique_intent):
        unique_intent = ['exchange_rate', 'contactless_not_working', 'declined_cash_withdrawal', 'card_arrival', 'card_payment_fee_charged', 'wrong_exchange_rate_for_cash_withdrawal', 'why_verify_identity', 'passcode_forgotten', 'cash_withdrawal_charge', 'top_up_limits', 'balance_not_updated_after_cheque_or_cash_deposit', 'transfer_timing', 'balance_not_updated_after_bank_transfer',
        'card_payment_not_recognised', 'failed_transfer', 'transaction_charged_twice', 'order_physical_card', 'wrong_amount_of_cash_received', 'card_not_working', 'pending_transfer', 'direct_debit_payment_not_recognised', 'getting_virtual_card', 'edit_personal_details', 'compromised_card', 'transfer_fee_charged', 'verify_my_identity', 'country_support', 'top_up_by_card_charge', 'Refund_not_showing_up', 'cancel_transfer', 'get_physical_card', 'receiving_money', 'card_swallowed', 'age_limit', 'extra_charge_on_statement', 'disposable_card_limits', 'change_pin', 'declined_card_payment',
        'card_delivery_estimate', 'reverted_card_payment?', 'card_payment_wrong_exchange_rate', 'get_disposable_virtual_card', 'terminate_account', 'pending_cash_withdrawal', 'top_up_reverted', 'transfer_not_received_by_recipient', 'supported_cards_and_currencies', 'lost_or_stolen_phone', 'category', 'exchange_via_app', 'atm_support', 'pending_card_payment', 'exchange_charge', 'lost_or_stolen_card', 'unable_to_verify_identity', 'getting_spare_card', 'virtual_card_not_working', 'cash_withdrawal_not_recognised', 'declined_transfer', 'top_up_by_cash_or_cheque', 'apple_pay_or_google_pay',
        'visa_or_mastercard', 'beneficiary_not_allowed', 'activate_my_card', 'pending_top_up', 'transfer_into_account', 'card_about_to_expire', 'top_up_failed', 'pin_blocked', 'verify_top_up', 'request_refund', 'card_linking', 'automatic_top_up', 'top_up_by_bank_transfer_charge', 'verify_source_of_funds', 'topping_up_by_card', 'card_acceptance', 'fiat_currency_support']
        pred = self.predictions(text, word_tokenizer, model_intent)
        result = self.get_final_output(pred, unique_intent)
        return result