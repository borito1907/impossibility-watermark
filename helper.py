import json
import openai
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken
import random
from time import sleep
import jsonlines
import re

DEF_MODEL = "gpt-4"
MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TOKENIZERS  = {model : tiktoken.encoding_for_model(MODELS[model]) for model in MODELS }
load_dotenv(dotenv_path='./.env') # take environment variables from .env with OPENAI_API_KEY=<your_key_here>
if os.getenv("OPENAI_API_ENDPOINT"):
    openai.api_base = os.getenv("OPENAI_API_ENDPOINT")
openai.api_key = os.getenv("OPENAI_API_KEY")

def set_seed(seed):
   random.seed(seed)
   np.random.seed(seed)

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    distance = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1      
    return distance

def tokens(s, model = DEF_MODEL):
  """Returns tokens of a string.
     Returns two lists, one of strings and the other of integers."""
  tokenizer = TOKENIZERS[model]
  L=tokenizer.encode(s)
  return [str(tokenizer.decode_single_token_bytes(t))[2:-1] for t in L],L

def count_tokens(s, tokenizer, model = DEF_MODEL):
  """Count the number of tokens in a string"""
  return len(tokenizer.encode(s))

def truncate(s,n, model = DEF_MODEL):
  """Truncase to n tokens"""
  tokenizer = TOKENIZERS[model]
  L = tokenizer.encode(s)
  return tokenizer.decode(L[:n])

def tokens2str(tokens, model = DEF_MODEL):
  tokenizer = TOKENIZERS[model]
  """Returns string from tokens (should get the integer tokens)"""
  return tokenizer.decode(tokens)

def read_jsonl(file: str):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def query_openai(prompt, model="text-davinci-003", max_tokens=512):
    # prompt = instruction+"\n"+query
    response = openai.Completion.create(
        engine=model, # "gpt-3.5-turbo-instruct"
        prompt=prompt,
        temperature=.2,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # stop=["\n"]
    )
    return response.choices[0].text

def chopped(s,k=30):
  """Chop a string to a shorter representation for prining"""
  if len(s)<=2*k: return(s)
  return f"{s[:k]}...{s[-k:]}"

def chat(message, history = [{"role": "system", "content": "You are a research assistant."}],
         model = "gpt-4", # model is "gpt-3" or "gpt-4"
         return_more = False,  # return also history and response object
         debug=True,  # print message and response
         supress_exception = False,  # supress exception and return None
         retries = 500, # number of times to retry if there is an exception
         tokenizer = None,
         **extra_params # extra parameters for Chat
         ):
  """General routine to send a message to GPT.
     Can take an optional parameter history of messages, and can also return message and history as extra parameter"""
  CONTEXT = {"gpt-4":8192, "gpt-3.5": 4096}
  if tokenizer is None:
    tokenizer = TOKENIZERS[model]
  hist_tokens  = count_tokens(", ".join(D["content"] for D in history), tokenizer)
  message_tokens = count_tokens(message, tokenizer)

  while retries >= 0:
    try:
      if debug: print(f"Message:\n {chopped(message)} ({message_tokens} tokens, history {hist_tokens} tokens)", flush=True)
      history = history + [{"role": "user", "content": f"{message}"}]
      params = dict(
            model = MODELS[model],
            messages= history,
            max_tokens=1024, # 512
            n=1,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            )
      params.update(extra_params) # update based on any extra parameters to add
      response = openai.ChatCompletion.create(**params)
      break
    except Exception as e:
      print(f"Error!:\n{e}")
      if retries:
        print(f"Retrying: {retries} tries left")
        sleep(1)
        retries -= 1
      elif not supress_exception:
        raise e
      else:
        return None

  text_response =  response.choices[0]['message']['content']
  if debug: print(f"Response:\n {chopped(text_response)} ({count_tokens(text_response, tokenizer)} tokens)", flush=True)
  if return_more:
    return text_response, history + [{"role": "assistant", "content": text_response}], response
  return text_response
    
def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    pattern = re.compile(r"<extra_id_\d+>")
    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills

def join_tokens(tokens):
    joined = " ".join(tokens)
    # Remove spaces before certain punctuation marks
    joined = re.sub(r'\s([,.;!?])', r'\1', joined)
    return joined

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [join_tokens(x) for x in tokens]
    return texts

def load_data(jsonl_file='data/lfqa/lfqa_umd.jsonl'):
    data = []
    with jsonlines.open(jsonl_file, 'r') as reader:
        for item in reader:
            data.append(item)
    return data