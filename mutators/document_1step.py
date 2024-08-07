import random
import nltk
from nltk.tokenize import sent_tokenize
import guidance
from guidance import models, gen, select, user, assistant
import hydra
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

class DocumentMutator_1step:  
    # NOTE: This current implementation is slow (~300 seconds) and must be optimized before use in the attack. 
    # One idea would be to have it suggest the edits in some structured format and then apply them outside of generation. 
    # This prevents it from having to copy / paste over the bulk of the response unchanged. 
    def __init__(self, cfg, llm = None) -> None:
        self.cfg = cfg
        self.llm = self._initialize_llm(llm)

        # Check if NLTK data is downloaded, if not, download it
        self._ensure_nltk_data()

    def _initialize_llm(self, llm):
        if not isinstance(llm, (models.Transformers, models.OpenAI)):
            log.info("Initializing a new Mutator model from cfg...")
            if "gpt" in self.cfg.model_id:
                llm = models.OpenAI(self.cfg.model_id)
            else:
                llm = models.Transformers(
                    self.cfg.model_id, 
                    echo=False,
                    cache_dir=self.cfg.model_cache_dir, 
                    device_map=self.cfg.device_map
                )
        return llm

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt') 

    def mutate(self, text):
        output = self.llm + paraphrase(text)
        print(f"output: {output}")
        return {
            "paraphrase_text": output["paraphrase_text"],
        }


@guidance
def paraphrase(lm, text):
    with user():
        lm += f"""\
        ### Task Description: 
        1. Carefully read the original text. 
        2. Paraphrase the original text so that it preserves the meaning and quality while varying the underlying words and sentence structures. 

        ### The original text: 
        {text}
        """
    with assistant():
        lm += f"""\
        ### Paraphrased text:
        ```json
        {{
            "paraphrase_text": "{gen('paraphrase_text', max_tokens=len(text.split()) * 1.5)}",
        }}```"""
    return lm


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def test(cfg):
        import time
        from utils import diff
        import textwrap

        print(f"Starting mutation...")

        text = textwrap.dedent("""
            Power is a central theme in J.R.R. Tolkien's The Lord of the Rings series, as it relates to the characters' experiences and choices throughout the story. Power can take many forms, including physical strength, political authority, and magical abilities. However, the most significant form of power in the series is the One Ring, created by Sauron to control and enslave the free peoples of Middle-earth.
            The One Ring represents the ultimate form of power, as it allows its possessor to dominate and rule over the entire world. Sauron's desire for the Ring drives much of the plot, as he seeks to reclaim it and use its power to enslave all of Middle-earth. Other characters, such as Gandalf and Frodo, also become obsessed with the Ring's power, leading them down dangerous paths and ultimately contributing to the destruction of their own kingdoms.
            Throughout the series, Tolkien suggests that power corrupts even the noblest of beings. As Gandalf says, "The greatest danger of the Ring is the corruption of the bearer." This becomes manifest as the characters who possess or covet the Ring become increasingly consumed by its power, losing sight of their original goals and values. Even those who begin with the best intentions, like Boromir, are ultimately undone by the temptation of the Ring's power.
            However, Tolkien also suggests that true power lies not in domination but in selflessness and sacrifice. Characters who reject the idea of using power solely for personal gain or selfish reasons are often the most effective in resisting the darkness of the Ring. For example, Aragorn's refusal to claim the throne or Sauron's rightful place as the Dark Lord illustrates this point. Instead, they embrace a more altruistic view of power, recognizing the importance of serving others and doing good.
            In conclusion, the One Ring symbolizes the corrosive nature of power while highlighting the potential for redemption through selflessness and sacrifice. Through the characters of the Lord of the Rings series, Tolkien demonstrates the various forms of power and their effects on individuals and society. He shows that the pursuit of power for personal gain can lead to corruption, but that true power emerges when one puts the needs of others first.
        """)

        text_mutator = DocumentMutator_1step(cfg.mutator_args)

        start = time.time()
        mutated_output = text_mutator.mutate(text)
        mutated_text = mutated_output["paraphrase_text"]
        delta = time.time() - start

        print(f"Original text: {text}")
        print(f"Mutated text: {mutated_text}")
        print(f"Diff: {diff(text, mutated_text)}")
        print(f"Time taken: {delta}")

    test()