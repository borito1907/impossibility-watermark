from transformers import pipeline
import random
import re
import string
import difflib
import torch
import logging

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # NOTE: Currently does not use GPU
        self.fill_mask = pipeline(
            "fill-mask", 
            model=self.model_name, 
            tokenizer=self.model_name,
            device=self.device
        )
        self.tokenizer_kwargs = {"truncation": True, "max_length": 512}

    def get_words(self, text):
        # Use a more comprehensive regex to capture more types of trailing punctuation
        m = re.match(r'^(.*?)([\.!?,:;()-]+)?$', text)
        core_text, end_punctuation = m.groups() if m else (text, None)
        words = core_text.split()
        return words, end_punctuation

    def select_random_segment(self, words):
        if len(words) <= self.max_length:
            return words, 0, len(words)
        start_index = random.randint(0, len(words) - self.max_length)
        return words[start_index:start_index + self.max_length], start_index, start_index + self.max_length

    def mask_random_word(self, words):
        if not words:  # Return the original text if there are no words to mask
            return words, None

        found_nice_word = False

        while not found_nice_word:
            index_to_mask = random.randint(0, len(words) - 1)  # Select a random index to mask
            word_to_mask = words[index_to_mask]  # Get the word at the selected index

            if not is_bullet_point(word_to_mask):
                found_nice_word = True

        # Create masked text by replacing only the selected word
        masked_text = ' '.join('<mask>' if i == index_to_mask else word for i, word in enumerate(words))
        return masked_text, word_to_mask

    def get_highest_score_index(self, suggested_replacements, blacklist):
        # Filter out dictionaries where 'token_str' is a blacklisted word
        filtered_data = [d for d in suggested_replacements if d['token_str'].strip().lower() not in blacklist]

        # Find the index of the dictionary with the highest score
        if filtered_data:
            highest_score_index = max(range(len(filtered_data)), key=lambda i: filtered_data[i]['score'])
            return filtered_data[highest_score_index]
        else:
            return suggested_replacements[0]

    def mutate(self, text, num_replacements=0.01):
        words, end_punctuation = self.get_words(text)

        if len(words) > self.max_length:
            segment, start, end = self.select_random_segment(words)
        else:
            segment, start, end = words, 0, len(words)

        if num_replacements < 0:
            raise ValueError("num_replacements must be larger than 0!")
        if 0 < num_replacements < 1:
            num_replacements = max(1, int(len(segment) * num_replacements))

        # log.info(f"Making {num_replacements} replacements to the input text segment.")

        replacements_made = 0
        while replacements_made < num_replacements:
            masked_text, word_to_mask = self.mask_random_word(segment)
            candidates = self.fill_mask(masked_text, top_k=3, tokenizer_kwargs=self.tokenizer_kwargs)
            suggested_replacement = self.get_highest_score_index(candidates, blacklist=[word_to_mask.lower()])

            log.info(f"word_to_mask: {word_to_mask}")
            log.info(f"suggested_replacement: {suggested_replacement['token_str']} (score: {suggested_replacement['score']})")

            # TODO: Move this above the log.
            if len(suggested_replacement['token_str'].strip()) == 0:
                continue
            
            left, word, right = strip_punct(word_to_mask)
            suggested_replacement['token_str'] = left + suggested_replacement['token_str'] + right

            log.info(f"actual suggested replacement: {suggested_replacement['token_str']}")
            
            segment = suggested_replacement['sequence'].split()
            replacements_made += 1

        # TODO: The last comma is still getting removed. Diagnose the issue and fix it.
        if end_punctuation:
            segment[-1] += end_punctuation

        log.info(words[end:])

        combined_text = ' '.join(words[:start]) + ' ' + ' '.join(segment) + ' ' + ' '.join(words[end:])
        return self.cleanup(combined_text)

    def cleanup(self, text):
        return text.replace("<s>", "").replace("</s>", "")

    def diff(self, text1, text2):
        # Splitting the texts into lines as difflib works with lists of lines
        text1_lines = text1.splitlines()
        text2_lines = text2.splitlines()
        
        # Creating a Differ object
        d = difflib.Differ()

        # Calculating the difference
        diff = list(d.compare(text1_lines, text2_lines))

        # Joining the result into a single string for display
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

    mutated_text = text

    words, end_punct = text_mutator.get_words(mutated_text)

    log.info(f"Words: {words}")
    log.info(f"End Punct: {end_punct}")

    for _ in range(20):
        mutated_text = text_mutator.mutate(mutated_text)
    delta = time.time() - start

    log.info(f"Original text: {text}")
    log.info(f"Mutated text: {mutated_text}")
    log.info(f"Original == Mutated: {text == mutated_text}")
    # log.info(f"Diff: {text_mutator.diff(text, mutated_text)}")
    log.info(f"Time taken: {delta}")

if __name__ == "__main__":
    test()