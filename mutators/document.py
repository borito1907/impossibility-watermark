import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class DocumentMutator(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model,
            cache_dir="/data2/.shared_models/",
            device_map="auto"
        )
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        # self.model = self.model.to(torch.device('cuda:1'))
        # self.model.cuda()
        self.model.eval()

    def mutate(self, input_text, lex_diversity=60, order_diversity=0, prefix="", sent_interval=1, max_length=1024, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, max_length=max_length, tokenizer=self.tokenizer, stop_strings=[" * ", "////"], **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text.rstrip(" *").strip()

def test():

    import time
    from utils import diff
    import pandas as pd

    print(f"Starting mutation...")

    dataset = pd.read_csv("./data/WQE/dev.csv")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    n=10
    avg_time=0
    dataset = dataset.head(n) 
    text_mutator = DocumentMutator()
    for index, row in dataset.iterrows():
      text = row["response_a"]

      start = time.time()
      mutated_text = text_mutator.mutate(text)
      delta = time.time() - start

      print(f"Original text: {text}")
      print(f"Mutated text: {mutated_text}")
      print(f"Diff: {diff(text, mutated_text)}")
      print(f"Time taken: {delta}")
      avg_time += delta
    print(f"Average time: {avg_time/n}")

if __name__ == "__main__":
    test()