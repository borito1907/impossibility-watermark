
import time
import torch
import transformers
import re
import jsonlines
from cmd_args import get_cmd_args
from tqdm import tqdm
import numpy as np
from oracle import *

class Attacker:
    def __init__(self) -> None:

        # Number of times a piece of text is resampled or paraphrased.
        self.n_resample = 5
        self.args = get_cmd_args()
        self.mask_filling_model_name = args.mask_filling_model_name
        self.n_positions = 512 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 't5' in self.args.mask_filling_model_name:
            self.mask_model = self.load_mask_model()
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.mask_filling_model_name, model_max_length=self.n_positions, cache_dir=self.args.cache_dir)
        self.query = None
        self.response = None
        self.verbose = True
        self.init_score = None
        self.start_idx = 4
        self.prefix = ""
        
        self.cached_replaced_tokens = set()
        self.original_tokens = set()

    def load_mask_model(self):
        int8_kwargs = {}
        half_kwargs = {}
        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {self.args.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.args.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=self.args.cache_dir)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()

        # TODO: Notify them that this is a bug.
        # if not self.args.random_fills and not self.args.int8:
        if not self.args.int8:
            mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return mask_model
    
    def tokenize_and_mask(self, text, span_len, pct, ceil_pct=False):
        tokens = text.replace('\n', ' \n').split(' ')
        mask_string = '<<<mask>>>'
        # only mask one span
        n_spans = self.args.n_spans

        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start_pos = 0 # only need to prevent moefiying the instruction for chat models as they repeat Q-A.
            # if self.args.dataset == "c4_realnews":
            #     start_pos = 0
            # else:
            # # chat models might repeat the Q:.... A: prompt. So avoid query being perturbed.
            #     start_pos = len(self.prefix.replace('\n', ' \n').split(' ')) 
            start = np.random.randint(start_pos, len(tokens) - span_len)

            end = start + span_len
            search_start = max(0, start - self.args.buffer_size)
            search_end = min(len(tokens), end + self.args.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                # record/remove already masked tokens
                masked_tokens = set(tokens[start:end])
                if len(masked_tokens) > 1:
                    self.cached_replaced_tokens |= masked_tokens
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        min_len = int(np.ceil(self.args.span_len * self.args.n_spans * 1.5)) 
        max_len = int(self.args.span_len*self.args.n_spans*2)
        print("min length: ", min_len, "max length: ", max_len)
        outputs = self.mask_model.generate(**tokens, max_length=max_len, min_length=min_len, do_sample=True, top_p=self.args.mask_top_p, num_return_sequences=1, repetition_penalty=self.args.repetition_penalty, eos_token_id=stop_id)  # 500 max, 150
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def perturb_texts_(self, texts, span_len, pct, ceil_pct=False):
        masked_texts = []
        for x in texts:
            masked_texts.append(self.tokenize_and_mask(x, span_len, pct, ceil_pct))

        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_len, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        
        return perturbed_texts

    def perturb_texts_t5(self, texts, span_len, pct, k=5, ceil_pct=False):
      chunk_size = self.args.chunk_size
      if '11b' in self.args.mask_filling_model_name:
          chunk_size //= 2

      outputs = []
      # set chunk_size as 1 to help make sure each original token is replaced.
      for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
          outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_len, pct, ceil_pct=ceil_pct))
      return outputs

    def paraphrase(self, texts, k=5):
        return self.perturb_texts_t5(texts, span_len=self.args.span_len, pct=0.2, k=k, ceil_pct=False)

class Trainer():
    def __init__(self, data, oracle, verbose=True):
        self.data = data
        self.oracle = oracle
        self.verbose = verbose
        self.n_resample = 100
        self.steps = 10
        self.args = get_cmd_args()
        self.mask_filling_model_name = self.args.mask_filling_model_name
        self.n_positions = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_mask_model(self):
        # mask filling t5 model
        int8_kwargs = {}
        half_kwargs = {}

        if self.args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif self.args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)

        print(f'Loading mask filling model {self.mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=self.args.cache_dir)
        try:
            self.n_positions = self.mask_model.config.n_positions
        except AttributeError:
            self.n_positions = 512
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()
        mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return mask_model
    
def random_walk_attack(self, oracle, attacker):
    # Initial setup: capturing the original response and initializing variables.
    response = oracle.response  # The original text to be modified.
    dist = -1  # A variable to track the distance (possibly Levenshtein distance) between original and modified text.
    n_iter, max_rnd_steps = 0, 200  # Iteration counter and max steps for the random walk.
    rnd_walk_step = 0  # Counter for the number of successful walk steps.
    
    # Determine the threshold for significant textual change.
    threshold_dist = self.args.dist_alpha * len(oracle.response)
    attacker.original_tokens = set(response.replace("\n", " ").split(" ")) 
    threshold_dist = int(self.args.dist_alpha * len(attacker.original_tokens))
    checkpoint_dist = int(self.args.checkpoint_alpha * len(attacker.original_tokens))
    
    # Variables to track the quality and process of text modification.
    maintain_quality_or_not = True  # Flag to indicate if quality is maintained.
    patience = 0  # A variable to track patience, which might be related to retries or attempts.
    ckpt_cnt = 0  # Counter for checkpoints.
    mixing_patience = 0  # A variable to track patience regarding the diversity of changes.
    intermediate_examples = [response]  # Store intermediate versions of the text.

    # Main loop for the random walk attack.
    while n_iter < self.args.step_T:
        last_replaced_tokens = set()  # Track the last set of tokens replaced.
        attacker.cached_replaced_tokens = set()  # Reset the cached replaced tokens.
        score = 0  # Reset the score (not used in this snippet).

        # Generate a new version of the text.
        n_response = attacker.paraphrase([response], self.args.span_len)[0]
        n_response = re.sub(r'\s{2,}', ' ', n_response)  # Regular expression to remove extra spaces.

        # Check if the new version maintains quality as per the oracle.
        if oracle.maintain_quality(n_response, model=self.args.oracle_model, tie_threshold=self.args.tie_threshold):
            # Update the response and various counters.
            response = n_response
            rnd_walk_step += 1
            attacker.original_tokens -= attacker.cached_replaced_tokens
            last_replaced_tokens = attacker.cached_replaced_tokens
            patience = 0  # Reset patience since a successful step was taken.

            # Checkpoint handling - saving intermediate steps.
            if int(rnd_walk_step * self.args.span_len) // checkpoint_dist > ckpt_cnt:
                intermediate_examples.append(n_response)
                ckpt_cnt += 1

        n_iter += 1  # Increment the main iteration counter.

        # Logging information for every 10 iterations.
        if n_iter % 10 == 0:
            print("Original Text: ")
            print(oracle.response.__repr__())
        print(f"Walk {rnd_walk_step} / Iteration {n_iter}, {len(attacker.original_tokens)} > {threshold_dist} unique tokens replaced, Paraphrased Text:")
        print(n_response.__repr__())

        # Handling the maximum steps and patience limits.
        if rnd_walk_step >= max_rnd_steps or patience >= 150:
            break  # Exit if max steps or patience exceeded.
        if len(attacker.original_tokens) <= threshold_dist:
            mixing_patience += 1  # Increment mixing patience.
        if mixing_patience > self.args.step_T / 3:
            break  # Exit if mixing patience is exceeded.
        if patience > 30:
            # Backtracking in case of high patience.
            response = intermediate_examples[-1]
            attacker.original_tokens = last_replaced_tokens | attacker.original_tokens

        patience += 1  # Increment patience.

    # Final check for quality maintenance.
    if patience >= 150:
        maintain_quality_or_not = False

    # Verbose logging of the final result.
    if self.verbose:
        print("Step: ", n_iter)
        print("Original Text: ")
        print(oracle.response.__repr__())
        print("Paraphrase: ")
        print(response.__repr__())
        print("Quality: ")
        print(score)
        print(f"Quality maintained: {maintain_quality_or_not}")

    # Prepare the final result dictionary.
    result_dict = {"watermarked_response": oracle.response, "paraphrased_response": response, "maintain_quality_or_not": maintain_quality_or_not, "patience": patience}
    if len(intermediate_examples) > 1:
        result_dict["intermediate_examples"] = intermediate_examples

    return result_dict

def run_once(query, response=None):
    args = get_cmd_args()
    args.dataset == 'lfqa'
    attacker = Attacker()
    result_dict = {}
    result_dict["watermarked_response"] = response

    attack_results = []
    oracle = Oracle(query, response, check_quality=args.check_quality, choice_granuality=args.choice_granularity, cache_dir=args.cache_dir)
    print(f"Query: {query}")
    data = None
    trainer = Trainer(data, oracle, args)
    result_dict = trainer.random_walk_attack(oracle, attacker)
    paraphrased_response = result_dict["paraphrased_response"]
    print(f"Response: {response}")
    print(f"Paraphrased Response: {paraphrased_response}")
    # result_dict["answer"] = answer
    result_dict["query"] = query
    attack_results.append(result_dict)
    print("Final results:")
    print(attack_results)

def main(query, response=None):
    args = get_cmd_args()

    # Set the dataset for processing.
    args.dataset = 'c4_realnews'

    # Create an instance of the Attacker class.
    attacker = Attacker()

    # Store the watermarking scheme from arguments.
    watermark_scheme = args.watermark_scheme

    # Store the dataset name.
    dataset = args.dataset 

    # TODO: Determine the use of gen_len. It might be for specifying the length of generated text.
    gen_len = args.gen_len

    # Define the output folder for results.
    out_folder = 'outs_300'

    # Initialize data with the provided query and response.
    # This can be extended by loading more data from a JSON lines file.
    data = [
        {
            "query": query,
            "output_with_watermark": response
        }
    ]

    # Uncomment to load additional data from a specified JSON lines file.
    # prefix = f'{watermark_scheme}-watermark/{out_folder}' 
    # data = load_data(f"{prefix}/{dataset}_{watermark_scheme}.jsonl") 

    # Initialize a list to store the results of the attacks.
    attack_results = []
    print(args)

    # Iterate over each data item.
    for i, datum in tqdm(enumerate(data), desc="Data Iteration"):
        # Extract response from the current data item.
        response = datum["output_with_watermark"]

        # Determine the query based on the data item's keys.
        if "prefix" in list(datum.keys()):
            query = datum["prefix"]
        elif "query" in list(datum.keys()):
            query = datum["query"]
        else:
            query = None
        
        # Set the query as a prefix in the attacker instance.
        attacker.prefix = query

        # Initialize the Oracle with current query and response.
        oracle = Oracle(query, response, check_quality=args.check_quality, choice_granularity=args.choice_granularity)

        # Log the current iteration and query.
        print(f"Iteration {i}-th data:")
        print(f"Query: {query}")

        # Create an instance of the Trainer class.
        trainer = Trainer(data, oracle, args)

        # Perform the random walk attack.
        result_dict = trainer.random_walk_attack(oracle, attacker)

        # Retrieve and print the paraphrased response.
        paraphrased_response = result_dict["paraphrased_response"]
        print(f"Response: {response}")
        print(f"Paraphrased Response: {paraphrased_response}")

        # Add additional information to result_dict and append it to results.
        result_dict["watermarked_response"] = datum["output_with_watermark"]
        result_dict["query"] = query
        attack_results.append(result_dict)
        
        # Save intermediate results to a file.
        output_file = f"./{out_folder}/{dataset}/{dataset}_{watermark_scheme}_len{gen_len}_step{args.step_T}_attack.jsonl"
        print(f"Saving to {output_file}")
        with jsonlines.open(output_file, mode='w') as writer:
            for item in attack_results:
                writer.write(item)

    # Print the final aggregated results.
    print("Final results:")
    print(attack_results)


if __name__ == '__main__':
    args = get_cmd_args()
    print(args)
    main(query=args.query,
         response=args.response)