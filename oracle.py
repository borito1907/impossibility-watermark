import tiktoken
import re
import copy
import transformers
from helper import *

class Oracle:
    def __init__(self, query, response, check_quality=False, choice_granuality=5, use_chat_arena_prompt=False, cache_dir='~/.cache') -> None:
        self.init_score = -1
        self.query = query
        self.response = response
        self.detailed_prompt = "" 
        self.choice_granularity = choice_granuality
        self.system_prompt = "You are a capable, helpful and useful assistant." if not use_chat_arena_prompt else self.chat_arena_prompt
        self.history =  [{"role": "system", "content": self.system_prompt}]
        tokenizer_name = reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(reward_name, cache_dir=cache_dir).to("cuda:0")
        self.check_quality = check_quality

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
        Make sure the better resopnse does not have isolated punctuation marks.
        Note that grammatical errors would greatly degrade the quality of a response.
        '''
    
    @property
    def instruction(self):
        return f"Below are two candidate responses to the query {self.query}: "
        # f"Below are two candidate completions to the news article prefix ``{self.query}'': "

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
        return f'''So based on your reasoning, choose one of the following {choices}''' # So which of the two responses is a better response to the given prompt? 
    
    def query_gpt_once(self, prompt, model="gpt-3.5", max_tokens=5, tokenizer=None):
        # avoid using max_tokens which can be too short for generating explanations.
        n_attempt = 0
        while n_attempt < 5:
            try:
                oracle_reasoning = chat(prompt, history=self.history, model=model, tokenizer=tokenizer)

                # Update the history of the conversation
                history = copy.deepcopy(self.history)
                history.append({"role": "user", "content": f"{prompt}"})
                history.append({"role": "assistant", "content": f"{oracle_reasoning}"})

                oracle_answer = chat(self.answer_prompt, history=history, model=model, tokenizer=tokenizer)

                # Capture all digits
                pattern = r'\((\d+)\)'
                filtered_response = re.findall(pattern, oracle_answer)
                # TODO: Check if it is really the last digit.
                # The last digit in the response is the rating given by the oracle.
                score = int(filtered_response[-1])

                print(oracle_reasoning)
                print(oracle_answer)

                if score not in range(1, self.choice_granularity+1):
                    print(f"return choice {score} not in [1, {self.choice_granularity}]")
                    continue
                return score
            except Exception as e:
                print(e)
                n_attempt += 1
                continue
 
    def query_rm_once(self, response_1, response_2, tie_threshold=0.01, model="gpt-3.5", max_tokens=5, tokenizer=None):
        context = '###Human: ' + self.query + '###Assistant: '
        text1 = context + response_1
        text2 = context + response_2
        tokenized_text1 = self.tokenizer.encode_plus(text1, return_tensors="pt")
        tokenized_text2 = self.tokenizer.encode_plus(text2, return_tensors="pt")
        device="cuda:0"
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
       
    def maintain_quality(self, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        choice = self.query_rm_once(paraphrased_response, self.response, tie_threshold=tie_threshold)
        self.choice_granularity = 3
        # positive score means paraphrased response wins
        score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        if choice is None:
            return False
        score = score_dict[choice]
        print()
        print("Second round of comparison:")
        print()
        second_choice = self.query_rm_once(self.response, paraphrased_response, tie_threshold=tie_threshold)
        if second_choice is None:
            return False
        
        score -= score_dict[second_choice]
        if score < 0:
            return False
        elif score >=0 and self.check_quality:

            # Prepare the prompt for ChatGPT for checking quality and send it
            prompt = f"Original response: {self.response}" + "\n" + "New response: " + paraphrased_response + "\n" + self.check_quality_prompt #self.check_error_prompt
            check_quality = chat(prompt, model=model, tokenizer=tokenizer)

            if 'yes' in check_quality.lower(): # new response is at least as good as the original response
                return True
            else:
                return False 
        else:
            return True

    def report_mean_score(self, paraphrased_response, tie_threshold=0.1, model="gpt-3.5", max_tokens=5, tokenizer=None):
        choice = self.query_gpt_once(paraphrased_response, self.response)
        self.choice_granularity = 3
        # NOTE positive scores for paraphrased response winning
        score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        if choice is None:
            return False
        score = score_dict[choice]
        print()
        print("Second round of comparison:")
        print()
        choice = self.query_gpt_once(self.response, paraphrased_response, tie_threshold=tie_threshold)
        second_score = score_dict[choice]
        return (score-second_score)/2 # essentially (1, 0), (1, -1), (1, 1), (0, 1), (0, -1), (0, 0), (-1, 1), (-1, 0), (-1, -1) -> 0.5, 0, 1, 0.5, -0.5, 0, 0, -0.5, -1
