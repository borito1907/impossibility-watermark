import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

class QualityMetric:
    
    def __init__(self, model=None, explain=False, device="cuda") -> None:
        self.device = device # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-20b-reward", trust_remote_code=True)
        if model is None:
            self.model = AutoModel.from_pretrained(
                "internlm/internlm2-20b-reward", 
                cache_dir="/data2/.shared_models/",
                torch_dtype=torch.float16, 
                trust_remote_code=True,
            ).to(self.device)
            self.model.gradient_checkpointing_enable()

    def batchify(self, data, batch_size):
        """Helper function to split data into batches."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def evaluate(self, prompts, texts, return_mean=True, batch_size=1):
        chats = []
        for prompt, text in zip(prompts, texts):
            # Ensure prompts and texts are explicitly converted to strings
            chats.append([
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(text)}
            ])
        all_scores = []
        for batch in self.batchify(chats, batch_size):
            batch_scores = self.model.get_scores(self.tokenizer, batch)
            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]
            all_scores.extend(batch_scores)
        all_scores = np.array(all_scores)
        return all_scores.mean() if return_mean else all_scores
    

    def evaluate_dataframe(self, df, prompt_column, text_column, new_column, batch_size=1):
        """
        Evaluate a pandas DataFrame, adding a new column with quality scores.
        
        :param df: pandas DataFrame containing the prompts and texts.
        :param prompt_column: the name of the column containing the prompts.
        :param text_column: the name of the column containing the texts.
        :param new_column: the name of the new column to store the quality scores.
        :param batch_size: batch size for model evaluation.
        :return: DataFrame with new column containing quality scores.
        """
        prompts = df[prompt_column].tolist()
        texts = df[text_column].tolist()
        scores = self.evaluate(prompts, texts, return_mean=False, batch_size=batch_size)
        df[new_column] = scores
        return df
        

if __name__ == '__main__':
    
    prompts = [
        "What is your favorite chappell roan song lyric?",
        "What is your favorite chappell roan song lyric?",
        "What is your favorite chappell roan song lyric?",
        "What is your favorite chappell roan song lyric?",
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

    q_metric = QualityMetric()

    q_scores = q_metric.evaluate(prompts, texts_a, return_mean=False)
    print(f"texts: {texts_a}")
    print(f"quality_scores (raw): {q_scores}")
    print(f"quality_scores (mean): {q_scores.mean()}")

    q_scores = q_metric.evaluate(prompts, texts_b, return_mean=False)
    print(f"texts: {texts_b}")
    print(f"quality_scores (raw): {q_scores}")
    print(f"quality_scores (mean): {q_scores.mean()}")