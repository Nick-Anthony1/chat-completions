from flask import Flask, render_template, request
from services.chat_completions_service import ChatConversation
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
main_conversation = ChatConversation()

# Define a route to render the basic chat page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle incoming chat messages
@app.route('/chatbot', methods=['POST'])
def chat():
    question = request.form['question']
    try:
        response = main_conversation.ask_question(question)
        return {"question": question, "response":response}
    
    except ValueError:
        return {"question": question, "response": "Sorry, I can't process that many words."}
    

if __name__ == '__main__':
    
    app.run(debug=True)

    app.run()