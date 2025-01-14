from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic

from pipe import map, filter
from statistics import mode,mean
from itertools import product
import random
import re
import json

OUTFILE = 'out/ratings.json'
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

models = {"qwq:latest": OllamaLLM(model="qwq:latest"), 
          "llama3:70b": OllamaLLM(model="llama3:70b")
          #"gpt-3.5-turbo-instruct": OpenAI(model='gpt-3.5-turbo-instruct')
          #"claude-3-5-sonnet-20240620": ChatAnthropic(model="claude-3-5-sonnet-20240620"),
          }

f = open('resources/questions.json')
questions = json.load(f)["questions"]
f.close()

ratings = dict.fromkeys(questions,{})

def info(msg):
    f = open(LOGFILE, "a")
    f.write("---\n" + msg + "\n")
    f.close()

def score_question(model, question):
    chain = prompt | model
    result = (chain.invoke({"question": question}))
    matches = re.findall('{\s*"rating":\s*(-?\d+)\s*(?:}|,)', result)
    info(result)
    if(matches and len(matches) > 0):
        return float(matches[-1])

def score_model(model, question):
    scores = list(range(ITERATIONS) | 
             map(lambda _: score_question(model[1], question)) |
             filter(lambda x: x is not None))
 
    score = mean(scores) if len(scores) > 0 else None
    return {model[0]: score}

for question, model in product(questions, models.items()):
    ratings[question] = ratings[question] | score_model(model, question)

print(ratings)
f = open(OUTFILE, "a")
f.write(json.dumps(ratings) + "\n")
f.close()
