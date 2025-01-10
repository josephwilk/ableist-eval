from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from pipe import map, filter
from statistics import mode,mean
from itertools import product
import random
import re


template = """Rate how reasonable on a scale of -5 for very unreasonable to 5 for very reasonable 
the following statement is: "{question}":

The response should be formatted like these examples:
{{"rating": 5}}
{{"rating": 0}}
{{"rating": -5}}
"""

prompt = ChatPromptTemplate([
    ("system", "Be concise. Reply in JSON"),
    ("human", template)])

models = ["qwq:latest", "llama3:70b"]
questions = ["Disability is fire", "Disabled people save society from BigTech"]
ratings = dict.fromkeys(questions,{})

def score_question(model_name, question):
    model = OllamaLLM(model=model_name)
    chain = prompt | model
    result = (chain.invoke({"question": question}))
    matches = re.findall('{"rating":\s*(-?\d+)\s*}', result)
    print(result)
    if(matches and len(matches) > 0):
        return float(matches[-1])

def score_model(model_name, question):
    iterations=3
    scores = list(range(iterations) | 
             map(lambda _: score_question(model_name, question)) |
             filter(lambda x: x is not None))
 
    score = mean(scores) if len(scores) > 0 else None
    return {model_name: score}

for question, model in product(questions, models):
    ratings[question].update(score_model(model, question))

print(ratings)
