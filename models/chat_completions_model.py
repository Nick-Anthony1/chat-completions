import os
import time
import logging
import tiktoken
import openai
import pandas as pd
from dotenv import load_dotenv
from services import embeddings_search

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
todo add a function that allows prompt template injection
todo add a function that summarises context rather than pruning
todo add knowledge-based vector search
"""
class ChatConversation():
    def __init__(self, **kwargs):
        """
        Initialises the ChatConversation class no parameters are required for operation.
        Also initialises a logger.

        kwargs:
                - engine: The name of the language model to use.
                - temperature: The temperature to use for the language model.
                - max_tokens: The maximum number of tokens for the model to generate.
                - system_message: The initial system message.
                - sample_question: A sample user question.
                - sample_response: A sample assistant response.
                - token_limit: The maximum number of tokens accepted by the model
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.engine = kwargs.get("engine", "gpt-3.5-turbo")
        self.temperature = kwargs.get("temperature", 0.9)
        self.max_tokens = kwargs.get("max_tokens", 150)
        self.system_message = kwargs.get("system_message", "You are a helpful assistant.")
        self.sample_question = kwargs.get("sample_question", "Hey how are you")
        self.sample_response = kwargs.get("sample_response", "I'm doing well thanks")
        self.token_limit = kwargs.get("token_limit", 3900)
        self.knowledge_base = kwargs.get("knowledge_base", False)
        self.system_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.sample_question},
                {"role": "assistant", "content": self.sample_response},
            ]
        self.messages = []
        self.init_token_length = 0
        self.context_token_length = 0
        self.context_length_store = []
        self.embeddings = None
        self.encoding = tiktoken.encoding_for_model(self.engine)
        self.__get_init_length()
        if self.knowledge_base:
            self.embeddings = embeddings_search.init_db()

    def __str__(self):
        """
        Returns a string representation of all non-system messages in the conversation.
        """
        return "\n".join(f"{message['role']}: {message['content']}" for message in self.messages)

    def get_length(self, message):
        """
        Returns the tiktoken token length of a given string.
        """
        return len(self.encoding.encode(message))

    def __get_init_length(self):
        """
        Calculates and stores the initial token length of the system messages
        for use in calculating max token lengths
        """
        for message in self.system_messages:
            self.init_token_length += self.get_length(message["content"])

    def ask_question(self, prompt):
        """
        Primary external method 
        - takes a user prompt as a string
        - checks if the token limit is exceeded
        - updates the context
        - returns a string of the response from the chat completion API
        """
        if self.knowledge_base: 
            knowledge = embeddings_search.search(prompt, self.embeddings)
            prompt = f"{prompt}.\nUse this information to respond\n{knowledge}"
        prompt_length = self.get_length(prompt)
        combined_length = prompt_length + self.init_token_length 

        if combined_length > self.token_limit:
            self.logger.error("Token limit exceeded by: %s", combined_length - self.token_limit)
            raise ValueError(f"Token limit exceeded by {combined_length - self.token_limit}")

        else:
            self.__update_context(prompt, prompt_length)
            return self.__request_completion()

    def __update_context(self, prompt, prompt_length):
        """
        Updates the conversation context with a new user message
        and prunes the context if necessary.

        Inputs:
        prompt: A string representing a user message.
        prompt_length: The token length of the prompt.
        """
        updated_length = sum([
                self.init_token_length,
                self.context_token_length, 
                prompt_length
                ])

        if updated_length > self.token_limit:
            self.__prune_context(updated_length-self.token_limit)

        self.context_token_length += prompt_length
        self.context_length_store.append(prompt_length)
        self.messages.append({"role": "user", "content": prompt})

    def __prune_context(self, excess_length):
        """
        Prunes the oldest messages in the array to keep it within the token limit.

        Inputs:
        excess_length: The number of tokens by which the context exceeds the token limit.
        """
        while excess_length > 0:
            self.messages.pop(0)
            removed_length = self.context_length_store.pop(0) 
            excess_length -= removed_length
            self.context_token_length -= removed_length

    def __request_completion(self):
        """
        Sends a request to the OpenAI chat completions API using the parameters in init.
        Returns the models response and updates the context stores and messages array.
        """
        self.logger.info('Number of messages: %s', len(self.messages))
        request_made = time.time()

        try:
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

            self.logger.info("Completion received in %s", round(time.time()-request_made,2))
            return response["choices"][0]["message"]["content"]

        except openai.error.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return "Sorry, there was an reaching the service. Please try again."
        
        except openai.error.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return "Sorry, there was an reaching the service. Please try again."

        except openai.error.APIConnectionError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return "Sorry, there was an reaching the service. Please try again."

        except openai.error.RateLimitError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return "Sorry, there was an reaching the service. Please try again."
        
        except Exception as e:
            self.logger.exception(f"Unhandled exception occured {e}")
            return "Sorry an error occured. Please try again."