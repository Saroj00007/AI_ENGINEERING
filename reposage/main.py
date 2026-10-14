from http import client
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


response = client.responses.create(
    model="gpt-5.5" , 
    instructions="you are helpful ai assistant that give answer in one word" , 
    input= " what is function in python"
)


print(response.output_text)