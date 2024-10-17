import random
import re
import string
import pandas as pd
import difflib
import torch
import logging
import os
from itertools import chain, zip_longest
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# This is from the Adaptive codebase.
def next_token_entropy(input_text, model, tokenizer, device):
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(next(model.parameters()).device)

        # print(f"Model is on device: {next(model.parameters()).device}")
        # print(f"Input IDs are on device: {input_ids.device}")

        outputs = model(input_ids)
        logits = outputs.logits.cpu()
        del outputs
        del input_ids
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
    return entropy

def compute_entropies(words, model, tokenizer, device):
    """
    Computes the entropy for each token in the input text.

    Args:
        words (list): List of words from the text.
        model: The language model used for entropy measurement.
        tokenizer: The tokenizer associated with the model.
        device: The device (CPU/GPU) to perform computations on.

    Returns:
        List[Tuple[int, float]]: A list of (index, entropy) tuples for each word.
    """
    entropy_scores = []
    for i, _ in enumerate(words):
        input_text = ' '.join(words[:i + 1])  # Use prefix up to current word
        entropy = next_token_entropy(input_text, model, tokenizer, device)
        entropy_scores.append((i, entropy))
    return entropy_scores


def compute_entropies_efficiently(text, model, tokenizer):
    """
    Computes the entropy of the next token probabilities for each token in the input text efficiently.

    Args:
        text (str): The input text.
        model: The language model used for entropy measurement.
        tokenizer: The tokenizer associated with the model.

    Returns:
        List[float], List[str]: A list of entropy values and corresponding tokens for each token position.
    """
    model.eval()
    with torch.no_grad():
        # Move input_ids to the same device as the model
        input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(model.device)
        
        # Get model outputs
        outputs = model(input_ids)
        logits = outputs.logits.cpu()  # shape: [1, seq_len, vocab_size]
        
        # Get tokens (excluding the last token, which doesn't have a next-token prediction)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])[:-1]
        
        # Now you can delete input_ids if necessary
        del outputs
        del input_ids
        
        # Compute probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)  # shape: [1, seq_len, vocab_size]
        
        # Compute entropy for each position (excluding the last token)
        entropies = []
        seq_len = logits.size(1)
        for t in range(seq_len - 1):
            prob_dist = probs[0, t, :]  # The model's prediction for the next token at position t+1
            entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))  # Add epsilon to avoid log(0)
            entropies.append(entropy.item())
        
        return entropies, tokens


def save_to_csv(data, file_path, rewrite=False):
    df_out = pd.DataFrame(data)
    
    # Ensure the directory exists
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    
    if os.path.exists(file_path) and not rewrite:
        df_out.to_csv(file_path, mode='a', header=False, index=False)  # Append without writing headers
    else:
        df_out.to_csv(file_path, index=False)  # Create new file with headers
    
    print(f"Data saved to {file_path}")

log = logging.getLogger(__name__)

def is_bullet_point(word):
    """
    Checks if the given word is a bullet point in the format '1.', '2.', etc.

    Args:
    word (str): The word to check.

    Returns:
    bool: True if the word is a bullet point, False otherwise.
    """
    # Regular expression pattern to match a digit followed by a period
    pattern = r'^\d+\.$'
    
    # Use re.match to check if the word matches the pattern
    return re.match(pattern, word) is not None

def strip_punct(word):
    """
    Strips punctuation from the left and right of the word and returns a tuple.

    Args:
    word (str): The word to process.

    Returns:
    tuple: A tuple containing the left punctuation, the stripped word, and the right punctuation.
    """
    if not word:  # If the word is empty, return an empty tuple
        return ("", "", "")
    
    # Initialize variables
    left_punctuation = ""
    right_punctuation = ""

    # Strip left punctuation
    i = 0
    while i < len(word) and word[i] in string.punctuation:
        left_punctuation += word[i]
        i += 1
    
    # Strip right punctuation
    j = len(word) - 1
    while j >= 0 and word[j] in string.punctuation:
        right_punctuation = word[j] + right_punctuation
        j -= 1
    
    # The stripped word
    stripped_word = word[i:j+1]

    return (left_punctuation, stripped_word, right_punctuation)

class WordMutator:
    def __init__(self, model_name="FacebookAI/roberta-large"):
        self.model_name = model_name
        self.max_length = 256
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.fill_mask = pipeline(
            "fill-mask", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=self.device
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

        self.measure_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
        self.measure_model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m", device_map="auto")
        self.measure_model.eval()

    def get_words(self, text):
        split = re.split(r'(\W+)', text)
        words = split[::2]
        punctuation = split[1::2]
        return words, punctuation

    def select_random_segment(self, words, punctuation):
        if len(words) <= self.max_length:
            # Ensure lengths match
            if len(punctuation) < len(words):
                punctuation.append('')
            return words, punctuation, 0, len(words)
        
        max_start = len(words) - self.max_length
        start_index = random.randint(0, max_start)
        end_index = start_index + self.max_length

        segment_words = words[start_index:end_index]
        segment_punct = punctuation[start_index:end_index]

        # Adjust punctuation length if necessary
        if len(segment_punct) < len(segment_words):
            segment_punct.append('')

        return segment_words, segment_punct, start_index, end_index

    def intersperse_lists(self, list1, list2):
        flattened = chain.from_iterable((x or '' for x in pair) for pair in zip_longest(list1, list2))
        return ''.join(flattened)
    
    def mask_word_using_candidate_indices(self, words, punctuation, candidate_indices):
        """
        Attempts to mask a word from the given list of candidate indices.
        If none of the candidates work, returns None.
        """
        if not words or not candidate_indices:
            return None, None, None

        random.shuffle(candidate_indices)

        found_nice_word = False
        index_to_mask = None

        attempt_count = 0
        max_attempts = len(candidate_indices)  # Limit attempts to provided candidates

        while not found_nice_word and attempt_count < max_attempts:
            # Use the provided candidate indices to try masking
            index_to_mask = candidate_indices[attempt_count]
            attempt_count += 1

            if index_to_mask < 0 or index_to_mask >= len(words):
                continue  # Skip invalid indices

            word_to_mask = words[index_to_mask]

            # Check if the selected word is valid (not a bullet point)
            if word_to_mask and not is_bullet_point(word_to_mask):
                found_nice_word = True

        # If no suitable word was found, return None
        if not found_nice_word:
            return None, None, None

        # Mask the selected word
        words_with_mask = words.copy()
        words_with_mask[index_to_mask] = self.fill_mask.tokenizer.mask_token

        # Reconstruct the masked text with interspersed punctuation
        masked_text = self.intersperse_lists(words_with_mask, punctuation)

        return masked_text, word_to_mask, index_to_mask

    def mutate(self, text, num_replacements=0.001):
        words, punctuation = self.get_words(text)

        if len(words) > self.max_length:
            segment, seg_punc, start, end = self.select_random_segment(words, punctuation)
        else:
            segment, seg_punc, start, end = words, punctuation, 0, len(words)
            # Ensure lengths match
            if len(seg_punc) < len(segment):
                seg_punc.append('')

        # Unpack the returned entropies and tokens
        entropies, tokens = compute_entropies_efficiently(text, self.measure_model, self.measure_tokenizer)

        # Create a list of (index, entropy) pairs
        entropy_scores = list(enumerate(entropies))

        log.info(f"Length of Entropy Scores: {len(entropy_scores)}")
        log.info(f"Length of Words: {len(words)}")

        # Filter entropy scores to only include those within the selected segment
        segment_entropies = [
            (i, entropy) for i, entropy in entropy_scores if start <= i < end
        ]

        # Adjust indices to be relative to the segment
        segment_entropies = [ (i - start, entropy) for i, entropy in segment_entropies ]

        # Sort by entropy in descending order and select the top 20 indices
        top_20_indices = sorted(segment_entropies, key=lambda x: x[1], reverse=True)[:20]
        candidate_indices = [index for index, _ in top_20_indices]

        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            num_replacements = max(1, int(len(segment) * num_replacements))

        log.info(f"Making {num_replacements} replacements to the input text segment.")

        replacements_made = 0
        while replacements_made < num_replacements:
            masked_text, word_to_mask, index_to_mask = self.mask_word_using_candidate_indices(segment, seg_punc, candidate_indices)

            if word_to_mask is None or index_to_mask is None:
                log.warning("No valid word found to mask!")
                break

            log.info(f"Masked word: {word_to_mask}")

            # Print the masked text for debugging
            log.info(f"Masked text: {masked_text}")

            # Ensure that the mask token is present
            if self.fill_mask.tokenizer.mask_token not in masked_text:
                log.warning("Mask token not found in masked_text")
                continue

            # Use fill-mask pipeline
            candidates = self.fill_mask(masked_text, top_k=3, tokenizer_kwargs=self.tokenizer_kwargs)

            if not candidates:
                log.warning("No candidates returned from fill-mask pipeline")
                continue

            suggested_replacement = self.get_highest_score_index(candidates, blacklist=[word_to_mask.lower()])

            # Ensure valid replacement
            if suggested_replacement is None or not re.fullmatch(r'\w+', suggested_replacement['token_str'].strip()):
                log.info(f"Skipping replacement: {suggested_replacement['token_str'] if suggested_replacement else 'None'}")
                continue

            log.info(f"word_to_mask: {word_to_mask}")
            log.info(f"suggested_replacement: {suggested_replacement['token_str']} (score: {suggested_replacement['score']})")

            # Replace the masked word in the segment using the index
            segment[index_to_mask] = suggested_replacement['token_str'].strip()
            replacements_made += 1

        # Ensure punctuation lengths match for reconstruction
        punct_before = punctuation[:start]
        if len(punct_before) < len(words[:start]):
            punct_before.append('')

        punct_after = punctuation[end:]
        if len(punct_after) < len(words[end:]):
            punct_after.append('')

        if len(seg_punc) < len(segment):
            seg_punc.append('')

        # Reconstruct the text
        combined_text = self.intersperse_lists(words[:start], punct_before) + \
                        self.intersperse_lists(segment, seg_punc) + \
                        self.intersperse_lists(words[end:], punct_after)

        return self.cleanup(combined_text)


    def get_highest_score_index(self, suggested_replacements, blacklist):
        filtered_data = [d for d in suggested_replacements if d['token_str'].strip().lower() not in blacklist]

        if filtered_data:
            highest_score_index = max(range(len(filtered_data)), key=lambda i: filtered_data[i]['score'])
            return filtered_data[highest_score_index]
        else:
            return None

    def cleanup(self, text):
        return text.replace("<s>", "").replace("</s>", "")

    def diff(self, text1, text2):
        text1_lines = text1.splitlines()
        text2_lines = text2.splitlines()
        d = difflib.Differ()
        diff = list(d.compare(text1_lines, text2_lines))
        diff_result = '\n'.join(diff)
        return diff_result


def test():

    import time
    import textwrap
    import os
   
    # text = textwrap.dedent("""
    #     Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
    #     The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
    #     Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
    #     However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
    #     In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
    # """)

    text = textwrap.dedent(""""What a fascinating individual! Let's dive into the psychological portrait of someone who values their time at an astronomical rate.

**Name:** Tempus (a nod to the Latin word for ""time"")

**Profile:**

Tempus is a highly successful and ambitious individual who has cultivated a profound appreciation for the value of time. Having made their fortune through savvy investments, entrepreneurial ventures, or high-stakes decision-making, Tempus has come to regard their time as their most precious asset.

**Core traits:**

1. **Frugal with time:** Tempus guards their schedule like Fort Knox. Every moment, no matter how small, is meticulously accounted for. They prioritize tasks with ruthless efficiency, ensuring maximum productivity while minimizing idle time.
2. **Opportunity cost obsession:** When making decisions, Tempus always weighs the potential benefits against the opportunity costs – not just in financial terms but also in terms of the time required. If a task doesn't yield substantial returns or value, it's quickly discarded.
3. **Time- compression mastery:** Tempus excels at streamlining processes, leveraging technology, and optimizing routines to minimize time expenditure. Their daily routine is honed to perfection, leaving room only for high-yield activities.
4. **Profound sense of self-worth:** Tempus knows their worth and won't compromise on it. They wouldn't dream of wasting their valuable time on menial tasks or engaging in unnecessary social niceties that don't provide tangible returns.

**Behavioral patterns:**

1. **Rapid-fire decision-making:** Tempus makes swift, calculated choices after considering the time investment vs. potential ROI (return on investment). This approach often catches others off guard, as they prioritize decisive action over protracted deliberation.
2. **Minimalist scheduling:** Meetings are kept concise, and phone calls are optimized for brevity. Tempus schedules meetings back-to-back to ensure their day is filled, leaving little room for casual conversation or chit-chat.
3. **Value-driven relationships:** Personal connections are assessed by their utility and alignment with Tempus' goals. Friendships and collaborations must yield tangible benefits or offer significant learning opportunities; otherwise, they may be relegated to a peripheral role or terminated.
4. **Unflinching efficiency:** When given a task or project, Tempus approaches it with laser-like focus, aiming to deliver results within minutes (or even seconds) rather than days or weeks. Procrastination is nonexistent in their vocabulary.

**Thought patterns and values:**

1. **Economic time value theory:** Tempus intuitively calculates the present value of future time commitments, weighing pros against cons to make informed decisions.
2. **Scarcity mentality:** Time is perceived as a finite resource, fueling their quest for extraordinary productivity.
3. **Meritocratic bias:** Tempus allocates attention based solely on meritocracy – i.e., people and ideas worth investing their time in have demonstrated excellence or promise substantial ROI.
4. **High-stakes resilience:** A true entrepreneurial spirit drives Tempus, embracing calculated risks and adapting rapidly to shifting circumstances.

**Weaknesses:**

1. **Insufficient relaxation and leisure:** Overemphasizing productivity might lead Tempus to neglect essential downtime, eventually causing burnout or exhaustion.
2. **Limited patience:** Tempus can become easily frustrated when forced to engage with inefficient or incompetent individuals.
3. **Difficulty empathizing:** Prioritizing logic and outcomes may sometimes cause them to overlook or misunderstand emotional nuances.
4. **Inflexible planning:** An unyielding adherence to precision-planned timetables could result in difficulties adjusting to unexpected setbacks or spontaneous opportunities.

Overall, Tempus embodies a fierce dedication to time optimization and resource management, fueled by unwavering confidence in their self-worth and capabilities.""")

    text_mutator = WordMutator()

    start = time.time()

    df = pd.read_csv('./data/WQE_adaptive/dev.csv')

    mutations_file_path = './inputs/word_mutator/test_1new.csv'

    for idx, row in df.head(50).iterrows():
        mutated_text = row['text']

        words, punct = text_mutator.get_words(mutated_text)

        log.info(f"Words: {words}")
        log.info(f"Punct: {punct}")

        # for _ in range(20):
        mutated_text = text_mutator.mutate(mutated_text)
        delta = time.time() - start

        log.info(f"Original text: {text}")
        log.info(f"Mutated text: {mutated_text}")
        log.info(f"Original == Mutated: {text == mutated_text}")
        # log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
        log.info(f"Time taken: {delta}")

        stats = [{'id': row.id, 'text': row.text, 'zscore' : row.zscore, 'watermarking_scheme': row.watermarking_scheme, 'model': row.model, 'gen_time': row.time, 'mutation_time': delta, 'mutated_text': mutated_text}]
        save_to_csv(stats, mutations_file_path, rewrite=False)

if __name__ == "__main__":
    test()