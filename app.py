from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
from collections import defaultdict
from flask import *
from flask_ngrok import run_with_ngrok
app = Flask(_name_)

# Your existing code for preprocessing and bot interaction
#function for preprocessing the datas
def preprocess_dataset(dataset):
    dialogs = dataset.split('\n\n')
    preprocessed_dialogs = []

    # Load the pre-trained model and tokenizer
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add the pad token
    model = AutoModelForCausalLM.from_pretrained('gpt2')

    for dialog in dialogs:
        user_utterances = dialog.split('\n')[::2]  # Extract user utterances
        bot_utterances = dialog.split('\n')[1::2]  # Extract bot utterances

        # Preprocess the user and bot utterances
        input_user_utterances = tokenizer(user_utterances, return_tensors='pt', truncation=True, padding=True)
        input_bot_utterances = tokenizer(bot_utterances, return_tensors='pt', truncation=True, padding=True)

        # Combine user and bot utterances
        preprocessed_dialog = '\n'.join([f'User: {u}\nBot: {b}' for u, b in zip(user_utterances, bot_utterances)])
        preprocessed_dialogs.append(preprocessed_dialog)

    return '\n\n'.join(preprocessed_dialogs)
def interact_with_bot(user_input):
    question = []
    answer = []
    lines = []
    with open('dialogs.txt', 'r') as f:
        lines = f.readlines()
        dialogs = []
        for line in lines:
            parts = line.split("\t")
            dialogs.append(parts)
            question.append(parts[0])
            answer.append(parts[1])
        print(len(question) == len(answer))
        qa_dict = {}

        # Populate dictionary
        for d in dialogs:
            question = d[0]
            answer = d[1]
            if question in qa_dict:
                qa_dict[question] = answer
            else:
                qa_dict[question] = answer

        # Chat function
        def chat(user_input):
            if user_input in qa_dict:
                return qa_dict[user_input]
            else:
                return "No answer"

        response = chat(user_input)
        return "Bot: " + response

#loading the dataset
def load_dataset():
    with open('dialogs.txt', 'r') as file:
        dataset = file.read()
    return dataset
# Load the dataset and preprocess it
dataset = load_dataset()
preprocessed_dataset = preprocess_dataset(dataset)
# Print the first dialog as the bot's initial response
first_dialog = preprocessed_dataset.split('\n\n')[0]
# Interact with the bot
interact_with_bot(preprocessed_dataset)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle user input and bot responses
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = interact_with_bot(user_input)
    return response

run_with_ngrok(app)
app.run()