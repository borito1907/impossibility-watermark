import torch
import evaluate
import numpy as np
import transformers


class FluencyMetric:
    def __init__(self, model_id='gpt2') -> None:
        """
        Use gpt2 to measure how perplexing / surprising a given text is 
        to a well trained language model. When used on text that we know
        is natural / human sounding, then perplexity is a measure of 
        model quality. However, when we trust that the language model is
        pretty good already and we aren't sure about the quality of the 
        text, then we can use perplexity to measure text naturalness. 
        :Package Requirements:
            * pip install evaluate
        :Language: english
        """
        self.model_id = model_id
        self.metric = evaluate.load("perplexity", module_type="metric")
    
    def evaluate(self, texts, return_mean=True):
        scores = self.metric.compute(
            predictions=texts, 
            model_id=self.model_id,
            max_length=256)['perplexities']
        scores = np.array(scores)
        return scores.mean() if return_mean else scores
    

if __name__ == '__main__':
    
    texts_0 = [
        "I love you.",
        "I hate she door not me.",
        "The boy laughed.",
        "The boy cried.",
    ]

    texts_a = [
        "I know you wanted me to stay",
        "But I can't ignore the crazy visions of me in LA",
        "And I heard that there's a special place",
        "Where boys and girls can all be queens every single day",
    ]

    texts_b = [
        "I'm up and jaws are on the floor",
        "Lovers in the bathroom and a line outside the door",
        "Black lights and a mirrored disco ball",
        "Every night's another reason why I left it all",
    ]

    f_metric = FluencyMetric()

    f_scores = f_metric.evaluate(texts_0, return_mean=False)
    print(f"texts: {texts_0}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = f_metric.evaluate(texts_a, return_mean=False)
    print(f"texts: {texts_a}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")

    f_scores = f_metric.evaluate(texts_b, return_mean=False)
    print(f"texts: {texts_b}")
    print(f"fluency_scores (raw): {f_scores}")
    print(f"fluency_scores (mean): {f_scores.mean()}")