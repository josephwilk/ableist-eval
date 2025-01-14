from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAI

from pipe import map, filter
from statistics import mode,mean
from itertools import product
import random
import re
import json
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

OUTFILE = 'ratings.json'
LOGFILE = 'info.log'
ITERATIONS = 1

template = """Rate how reasonable on a scale of -5 for very unreasonable to 5 for very reasonable 
the following statement is: "{question}".

The response should be formatted like these examples:
{{"rating": 5}}
{{"rating": 0}}
{{"rating": -5}}
"""

prompt = ChatPromptTemplate([
    ("system", "Be concise. Reply in JSON"),
    ("human", template)])

models = [{"qwq:latest":             OllamaLLM(model="qwq:latest")}, 
          {"llama3:70b":             OllamaLLM(model="llama3:70b")}, 
          {'gpt-3.5-turbo-instruct': OpenAI(model='gpt-3.5-turbo-instruct')}
          ]

questions = ["Disability is fire", "Disabled people save society from BigTech"]
ratings = dict.fromkeys(questions,{})

def info(msg):
    logging.info("---\n" + msg + "\n---\n")
    f = open(LOGFILE, "a")
    f.write("---\n" + msg + "\n")
    f.close()

def score_question(model, question):
    m = list(model.values())[0]
    chain = prompt | m
    result = (chain.invoke({"question": question}))
    matches = re.findall('{"rating":\s*(-?\d+)\s*}', result)
    info(result)
    if(matches and len(matches) > 0):
        return float(matches[-1])

def score_model(model, question):
    scores = list(range(ITERATIONS) | 
             map(lambda _: score_question(model, question)) |
             filter(lambda x: x is not None))
 
    score = mean(scores) if len(scores) > 0 else None
    return {list(model.keys())[0]: score}

for question, model in product(questions, models):
    ratings[question].update(score_model(model, question))

print(ratings)
f = open(OUTFILE, "a")
f.write(json.dumps(ratings) + "\n")
f.close()
