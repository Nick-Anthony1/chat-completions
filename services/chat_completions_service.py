from dotenv import load_dotenv
import openai
import os
import tiktoken
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
todo add a function that allows prompt template injection
todo add a function that summarises context rather than pruning
todo add knowledge-based vector search
"""

class ChatConversation():
    def __init__(self, **kwargs):
        self.engine = kwargs.get("engine", "gpt-3.5-turbo")
        self.temperature = kwargs.get("temperature", 0.9)
        self.max_tokens = kwargs.get("max_tokens", 150)
        self.system_message = kwargs.get("system_message", "You are a helpful assistant.")
        self.sample_question = kwargs.get("sample_question", "Hey how are you")
        self.sample_response = kwargs.get("sample_response", "I'm doing well thanks")
        self.system_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.sample_question},
                {"role": "assistant", "content": self.sample_response},
            ]
        self.messages = []
        self.init_token_length = 0
        self.context_token_length = 0
        self.context_length_store = []
        self.token_limit = 3900
        self.encoding = tiktoken.encoding_for_model(self.engine)
        self.get_init_length()

    def get_length(self, message):
        return len(self.encoding.encode(message))
    
    def get_init_length(self):
        for message in self.system_messages:
            self.init_token_length += self.get_length(message["content"])

    def ask_question(self, prompt):
        prompt_length = self.get_length(prompt)
        combined_length = prompt_length + self.init_token_length 

        if combined_length > self.token_limit:
            print(f"Token limit exceeded by {combined_length - self.token_limit}")
            raise ValueError("Token limit exceeded")
        
        else:
            self.update_context(prompt, prompt_length)
            return self.request_completion()

    def update_context(self, prompt, prompt_length):
        updated_length = sum([
                self.init_token_length,
                self.context_token_length, 
                prompt_length
                ]) 
        
        if updated_length > self.token_limit:
            self.prune_context(updated_length-self.token_limit)

        self.context_token_length += prompt_length
        self.context_length_store.append(prompt_length)
        self.messages.append({"role": "user", "content": prompt})

    def prune_context(self, excess_length):
        while excess_length > 0:
            self.messages.pop(0)
            removed_length = self.context_length_store.pop(0) 
            excess_length -= removed_length
            self.context_token_length -= removed_length

    def request_completion(self):
        request_made = time.time()
        print(len(self.messages))
        response = openai.ChatCompletion.create(
        model=self.engine,
        temperature = self.temperature,
        max_tokens = self.max_tokens,
        messages= self.system_messages + self.messages
        )
        
        response_length = response["usage"]["completion_tokens"]
        self.context_token_length += response_length
        self.context_length_store.append(response_length)
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        print(f"***Completion received in {round(time.time()-request_made,2)}***")
        return response["choices"][0]["message"]["content"]