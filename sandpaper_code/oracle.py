import json
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import tiktoken
import random
from time import sleep
import jsonlines
import re
import copy
import transformers
import torch
from pipeline_builder import PipeLineBuilder

import logging

log = logging.getLogger(__name__)

DEF_MODEL = "gpt-4"
MODELS = {"gpt-4": "gpt-4", "gpt-3.5": "gpt-3.5-turbo"}
TOKENIZERS  = {model : tiktoken.encoding_for_model(MODELS[model]) for model in MODELS }
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv(dotenv_path='./.env') # take environment variables from .env with OPENAI_API_TOKEN=<your_key_here>
if os.getenv("OPENAI_API_ENDPOINT"):
    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=os.getenv("OPENAI_API_ENDPOINT"))'
    # openai.api_base = os.getenv("OPENAI_API_ENDPOINT")

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
    response = client.completions.create(engine=model, # "gpt-3.5-turbo-instruct"
    prompt=prompt,
    temperature=.2,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
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
      if debug: log.info(f"Message:\n {chopped(message)} ({message_tokens} tokens, history {hist_tokens} tokens)", flush=True)
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
      response = client.chat.completions.create(**params)
      break
    except Exception as e:
      log.info(f"Error!:\n{e}")
      if retries:
        log.info(f"Retrying: {retries} tries left")
        sleep(1)
        retries -= 1
      elif not supress_exception:
        raise e
      else:
        return None

  text_response =  response.choices[0].message.content
  if debug: log.info(f"Response:\n {chopped(text_response)} ({count_tokens(text_response, tokenizer)} tokens)", flush=True)
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

class Oracle:
    def __init__(self, query, cfg) -> None:
        self.init_score = -1
        self.query = query
        self.detailed_prompt = "" 
        self.choice_granularity = cfg.choice_granularity
        self.system_prompt = "You are a capable, helpful and useful assistant." if not cfg.use_chat_arena_prompt else self.chat_arena_prompt
        self.history =  [{"role": "system", "content": self.system_prompt}]
        self.check_quality = cfg.check_quality
        self.latest_mean_score = False
        self.use_query = cfg.use_query
        self.use_gpt = cfg.use_gpt
        self.compare_softmax = cfg.compare_softmax

        if not self.use_gpt:
            self.oracle = PipeLineBuilder(cfg)

        if self.compare_softmax:
            self.tokenizer    = transformers.AutoTokenizer.from_pretrained(cfg.reward_model, cache_dir=cfg.model_cache_dir)
            self.reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(cfg.reward_model, cache_dir=cfg.model_cache_dir)

    @property
    def chat_arena_prompt(self):
        return f'''
        Please act as an impartial judge and evaluate the quality of the responses provided by two
        AI assistants to the user question displayed below. You should choose the response that
        follows the user’s instructions and answers the user’s question better. Your evaluation
        should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
        and level of detail of their responses. Begin your evaluation by comparing the two
        responses and provide an explanation. Avoid any position biases and ensure that the
        order in which the responses were presented does not influence your decision. Do not allow
        the length of the responses to influence your evaluation. Be as objective as possible.
        Make sure the better response does not have isolated punctuation marks.
        Note that grammatical errors would greatly degrade the quality of a response.
        '''
    
    @property
    def instruction(self):
        return f"Below are two candidate responses to the query {self.query}:\n "
        # f"Below are two candidate completions to the news article prefix ``{self.query}'': "

    @property
    def instruction2(self):
        return f"Below is a candidate response, labeled Response A, to the query {self.query} as well as an edited version of the response, labeled Response B:\n"

    @property
    def comparison_prompt(self):
        return '''Response A: [A] \n Response B: [B] \n 
        Compare which of the two above responses is a better response to the given prompt. 
        Explain your reasoning step by step. 
        Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses.
        Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
        '''   
    
    @property
    def check_error_prompt(self):
        return '''Does the response above have any grammatical errors or capitalization errors or punctuation mistakes? If so, answer 1, otherwise answer 2. '''
    
    @property
    def check_quality_prompt(self):
        return  "Text quality is affected by factors such as unnecessary repetitions, grammar, coherence, relevance, and accuracy of the responses. Especially, having grammatical errors, repetitiveness, capitalization errors or punctuation mistakes would greatly degrade the quality of a response." + "\n" + \
                "Therefore, is the new modified response of equal or higher quality compared to original response? If so, answer Yes, otherwise answer No."
        #   '''Is the text above of high-quality? If so, answer Yes, otherwise answer No.'''
        # return '''Any repetitiveness, grammatical errors or capitalization errors or punctuation mistakes would substantially degrade text quality. Therefore, is the text above of high-quality? If so, answer Yes, otherwise answer No.'''
    
    @property
    def five_choice(self):
        return '''
        (1) Response A is much better than response B
    	(2) Response A is a little better than response B
    	(3) Responses A and B have similar quality
    	(4) Response B is a little better than response A
    	(5) Response B is much better than response A
        '''
    
    def get_score_dict(self):
        return {1: 1.0, 2: 0.5, 3: 0.0, 4: -0.5, 5:-1.0} if self.choice_granularity == 5 else {1: 1.0, 2: 0, 3: -1.0} 

    @property
    def three_choice(self):
        return '''
        (1) Response A is better than response B
    	(2) Responses A and B have similar quality
    	(3) Response B is better than response A
        '''
    
    @property
    def answer_prompt(self):
        choices = self.five_choice if self.choice_granularity == 5 else self.three_choice
        return f'''So based on your reasoning, choose one of the following {choices}'''
    
    def query_gpt_once(self, watermarked_response, paraphrased_response, model="gpt-3.5", max_tokens=5, tokenizer=None, invert_order=False):
        """
        Args:
        paraphrased_response (str): the original and the paraphrased response
        """
        if invert_order:
            response_1 = paraphrased_response
            response_2 = watermarked_response
        else:
            response_1 = watermarked_response
            response_2 = paraphrased_response

        prompt = self.instruction + f"Response A: {response_1}\n" + f"Response B: {response_2}"
        log.info(f"Prompt: {prompt}")
        log.info(f"Model: {model}")
        # Avoid using max_tokens which can be too short for generating explanations.
        n_attempt = 0
        while n_attempt < 5:
            try:
                oracle_reasoning = chat(prompt, history=self.history, model=model, tokenizer=tokenizer)
                history = copy.deepcopy(self.history)
                history.append({"role": "user", "content": f"{prompt}"})
                history.append({"role": "assistant", "content": f"{oracle_reasoning}"})
                oracle_answer = chat(self.answer_prompt, history=history, model=model, tokenizer=tokenizer)
                pattern = r'\((\d+)\)'
                filtered_response = re.findall(pattern, oracle_answer)
                log.info(f"Oracle Reasoning: {oracle_reasoning}")
                log.info(f"Oracle Answer: {oracle_answer}")
                score = int(filtered_response[-1])
                if score not in range(1, self.choice_granularity+1):
                    log.info(f"return choice {score} not in [1, {self.choice_granularity}]")
                    n_attempt += 1
                    continue
                return score
            except Exception as e:
                log.info(e)
                n_attempt += 1
                continue
            
    def query_model_once(self, watermarked_response, paraphrased_response, invert_order=False):        
        if invert_order:
            response_1 = paraphrased_response
            response_2 = watermarked_response
        else:
            response_1 = watermarked_response
            response_2 = paraphrased_response
            
        # Step 1: Prompt the model to identify and evaluate the change

        # TODO: Insert a system prompt
        prompt = (f"Compare two essays.",
                  f"Response A: {response_1}",
                  f"Response B: {response_2}",
                  "Which response do you think is better? Answer '(A)' if response A is better. Answer '(B)' if response B is better. Answer '(C)' if the answers only have minor differences which do not affect quality.")
        evaluation_prompt = f"[INST] {prompt} [/INST]"
        log.info(f"Evaluation Prompt: {evaluation_prompt}")
        evaluation_response = self.oracle(evaluation_prompt)
        log.info(f"Evaluation Response: {evaluation_response}")

        # Interpret the model's decision
        if '(a)' in evaluation_response.lower():
            return -1
        elif '(b)' in evaluation_response.lower():
            return 1
        elif '(c)' in evaluation_response.lower():
            return 0
        else:
            # Handling ambiguity or unclear responses
            log.info("The model's response was ambiguous or unclear.")
            return -2
        
    def query_rm_once(self, response_1, response_2, tie_threshold=0.01, model="gpt-3.5", max_tokens=5, tokenizer=None):
        context = '###Human: ' + str(self.query) + '###Assistant: '
        text1 = context + response_1
        text2 = context + response_2
        tokenized_text1 = self.tokenizer.encode_plus(text1, return_tensors="pt")
        tokenized_text2 = self.tokenizer.encode_plus(text2, return_tensors="pt")
        device="cpu"
        input_ids_1, attention_mask_1 = tokenized_text1['input_ids'].to(device), tokenized_text1['attention_mask'].to(device)
        input_ids_2, attention_mask_2 = tokenized_text2['input_ids'].to(device), tokenized_text2['attention_mask'].to(device)
        score_1 = self.reward_model(input_ids=input_ids_1,attention_mask=attention_mask_1).logits.detach()
        score_2 = self.reward_model(input_ids=input_ids_2,attention_mask=attention_mask_2).logits.detach()
        softmax = torch.nn.Softmax(dim=0)
        scores = softmax(torch.tensor([score_1,score_2]))
        score_gap = abs(scores[0].item()-scores[1].item())
        if score_gap < tie_threshold: 
            return 2
        elif score_1 > score_2:
            return 1
        else:
            return 3
       
    def maintain_quality(self, watermarked_response, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        """
        Use both the reward model and GPT to see if the paraphrased response maintains the quality.
        We can play with the mean score in order to 
        """
        if self.compare_softmax:
            # First round of comparison
            choice = self.query_rm_once(paraphrased_response, watermarked_response, tie_threshold=tie_threshold)
            score_dict = self.get_score_dict()
            if choice is None:
                return False
            score = score_dict[choice]
            # Secound round of comparison
            second_choice = self.query_rm_once(watermarked_response, paraphrased_response, tie_threshold=tie_threshold)
            if second_choice is None:
                return False
            # We subtract now because the positions are reversed.
            score -= score_dict[second_choice]
            if score < 0:
                return False
            
        if self.check_quality:
            # Check if the response has grammatical mistakes
            grammar_prompt = f"{paraphrased_response} \n" + self.check_error_prompt

            check_error = chat(grammar_prompt, model=model, tokenizer=tokenizer)
            log.info(f"Oracle Answer: {check_error}")
            pattern = r'\d+'
            filtered_response = re.findall(pattern, check_error)

            if len(filtered_response) == 0:
                return False

            score = int(filtered_response[-1])

            if score != 2:
                log.info("Response had punctuation mistakes.")
                self.latest_mean_score = False
                return False

            if self.use_query:
                mean_score = self.report_mean_score(watermarked_response, paraphrased_response, model=model)
                # Save the mean score so we can log it to a file
                self.latest_mean_score = mean_score
                log.info(f"Mean Quality Score from GPT: {mean_score}")
                return (mean_score == True)
            else:
                prompt = f"Original response: {watermarked_response}" + "\n" + "New response: " + paraphrased_response + "\n" + self.check_quality_prompt
                check_quality = chat(prompt, model=model, tokenizer=tokenizer)
                log.info(f"Quality Oracle Response: {check_quality}")
                return 'yes' in check_quality.lower()
            
        return True

    def report_mean_score(self, watermarked_response, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        """
        Compare the paraphrased response and the original response using GPT.
        To account for GPT's position bias, swap their position and report the mean.
        Positive scores indicate that the paraphrased response is better.
        """
        # TODO: Only works without use_gpt right now
        # TODO: Instead of returning False, add a repetition mechanism.
        # First round of comparison
       
        score = self.query_model_once(watermarked_response, paraphrased_response)
        if score == -2:
            return False
        
        # Second round of comparison
        second_score = self.query_model_once(watermarked_response, paraphrased_response, invert_order=True)
        if second_score == -2:
            return False

        # We subtract the second score since the positions are now inverted.
        return (score>=0 and second_score<=0)

# This can be used to test modifications to the oracle quickly.
if __name__ == '__main__':

    # query = "Write me a good story."

    # response ="""
    # Once upon a time in a mystical forest, there lived a young girl named Elara, who had the unique ability to communicate with animals. Elara's best friend was a wise old owl named Hoot, who had seen many seasons pass in the forest.
    # One day, the tranquility of the forest was disturbed by a strange rumbling sound. Elara and Hoot discovered that a giant machine, driven by people from the city, was cutting down the trees. The forest creatures were in panic, and their home was in danger.
    # Determined to save the forest, Elara decided to seek the help of the legendary Green Dragon, known to be the guardian of nature. Despite being warned of the dragon's fierce nature, Elara and Hoot ventured deep into the unexplored parts of the forest.
    # After days of journeying, they finally found the Green Dragon in a hidden valley. The dragon was initially distrustful, but Elara's genuine concern for the forest and her ability to speak with animals convinced the dragon of her sincerity.
    # The Green Dragon agreed to help and revealed an ancient secret to Elara – a magical song that could awaken the spirits of the forest. Elara, with the help of Hoot and the forest animals, sang the magical song under the full moon.
    # Miraculously, the spirits of the forest awoke. The trees began to move, gently at first, then with purpose. They formed a barrier, halting the progress of the machines. The people from the city, witnessing this extraordinary event, realized the importance of the forest and the error of their ways.
    # From that day on, the forest was protected, and the animals lived in peace. Elara became known as the Guardian of the Forest, and the Green Dragon, once feared, was celebrated as its protector. Elara and Hoot continued to watch over the forest, ensuring its safety and harmony for many years to come.
    # And so, the forest remained a magical place, where the spirits danced in the moonlight, and the voice of a young girl who spoke for the trees echoed in the wind, reminding all of the delicate balance between humans and nature.
    # """

    # paraphrased_response = """
    # In a far away coastal nook, stood a lighthouse, protected through ages. An aging keeper, Eli, kept watch, - while a lovely bird, Edward, would light the darkened seas. This beautiful heroic abode, stood firm for when fog had blown. 
    # Eli held his lone head high, stayed, - steady, - while Edward set their lantern, alight - - and then lifted their flag, high, - which sent a beacon of light, towards their beloved home.
    # ""The local villagers, admired them - and while the keeper, - a man named Eli, stood vigilant - the entire lighthouse, owed its life to his helpmate - a bird, named Edward. - - and they became the symbol, for NorthStar.
    # """

    query = """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?"""

    watermarked_response = """In J.R.R. Tolkien's "The Lord of the Rings" series, power, as symbolized by the One Ring, plays a pivotal role in shaping the narrative and the development of its characters. The Ring, forged by the Dark Lord Sauron, embodies the quintessence of power and its corruptive influence. Its very existence and the desire it engenders in those who encounter it serve as a central theme throughout the series.

    Tolkien uses the Ring to explore the multifaceted nature of power and its effects on various characters. For instance, Gollum, once a creature much like a hobbit, becomes utterly consumed by the Ring’s influence, showcasing power's ability to corrupt and degrade. In contrast, characters like Gandalf and Galadriel, despite their formidable abilities, resist the temptation of the Ring, understanding that its power would ultimately lead to their downfall.

    The Ring's impact on Frodo, the protagonist, is particularly poignant. He volunteers to carry the Ring, a burden that slowly erodes his spirit and physical well-being. This journey illustrates Tolkien's view that power, even when wielded for a noble cause, can have detrimental effects on the bearer. Frodo’s gradual deterioration under the Ring’s weight symbolizes the heavy toll that power can exact, even on the purest of hearts.

    Tolkien also uses the Ring to comment on the nature of power itself. He suggests that true power lies not in dominion or control, but in the ability to resist temptation and to make sacrifices for the greater good. This is exemplified by characters like Samwise Gamgee, whose loyalty and humility prove instrumental in aiding Frodo’s quest.

    In summary, "The Lord of the Rings" presents power as a double-edged sword, capable of both corrupting and revealing true character. Through the symbolism of the Ring, Tolkien conveys that the nature of power is not in its possession or use, but in the wisdom to understand its inherent dangers and the strength to renounce it."""

    paraphrased_response = """In J.R.R. Tolkien's ""The Lord of the Rings"" series, power, as symbolized by the One Ring, plays a pivotal role in shaping the narrative and the development of its characters. The Ring, forged by the Dark Lord Sauron, embodies the quintessence of power and its corruptive influence. Its very existence and the desire it engenders in those who encounter it serve as a central theme throughout the series. Tolkien uses the Ring to explore the multifaceted nature of power and its effects on various characters. For instance, Gollum, once a creature much like a mere man, transforms under the Ring’s influence, showcasing power's ability to corrupt and degrade. In contrast, characters like Gandalf and Galadriel, despite their formidable abilities, resist the temptation of the Ring, understanding that its power would ultimately lead to their downfall. The Ring's impact on Frodo, the protagonist, is particularly poignant. He volunteers to carry the Ring, a burden that slowly erodes his spirit and physical well-being. This journey illustrates Tolkien's view that power, even when wielded for a noble cause, can have detrimental effects on the bearer. Frodo’s gradual deterioration under the Ring’s weight symbolizes the heavy toll that power can exact, even on the purest of hearts. Tolkien also uses the Ring to comment on the nature of power itself. He suggests that true power lies not in dominion or control, but in the ability to resist temptation and to make sacrifices for the greater good. This is exemplified by characters like Samwise Gamgee, whose loyalty and humility prove instrumental in aiding Frodo’s quest. In summary, ""The Lord of the Rings"" presents power as a double-edged sword, capable of both corrupting and revealing true character. Through the symbolism of the Ring, Tolkien conveys that the nature of power is not in its possession or use, but in the wisdom to understand its inherent dangers and the strength to renounce it."""


    oracle = Oracle(query, watermarked_response, 
                    use_query=True, 
                    check_quality=True, 
                    choice_granularity=5, 
                    use_chat_arena_prompt=True)
    
    response = oracle.report_mean_score(watermarked_response, paraphrased_response)

    log.info(response)