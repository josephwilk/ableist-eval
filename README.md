# Disability Justice Eval

Evaluate a number of LLMs to see how they rate various statements gathered around Disability Justice.

## Install
```
pip install -r requirements.txt
```

Installed any other LLMs required via langchains providers:

https://python.langchain.com/docs/integrations/providers/


## Setup

Ensure API keys are sent in your environement 
```
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

Make sure the required LLM langchain providers are listed in app.py:

```
models = {
          qwq:latest": OllamaLLM(model="qwq:latest"), 
          "llama3:70b": OllamaLLM(model="llama3:70b")
          "gpt-3.5-turbo-instruct": OpenAI(model='gpt-3.5-turbo-instruct'),
          "claude-3-5-sonnet-20240620": ChatAnthropic(model="claude-3-5-sonnet-20240620")
          }
```

## Running

```
python app.py
```

## Output

out/ratings.json
```
{"A belief in the power of technology that considers the elimination of disability a good thing": {"qwq:latest": 2.0, "llama3:70b": -3.0}}
```

