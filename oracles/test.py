# RUN: CUDA_VISIBLE_DEVICES=2,7 python -m oracles.test

import guidance
from guidance import models, gen, select, user, system, assistant

@guidance
def annotation_fn(lm, prompt):
    n="\n"
    with user():
        lm += f"""
        ### Task Description: 
        1. Carefully read the given prompt.
        2. Provide a topic heirarchy in the form of "high-level > mid-level > low-level".
        Examples:
            Science > Biology > Genetics
            Technology > Software Development > Machine Learning
            Health > Nutrition > Vitamins and Minerals
        2. If no topic is appropriate, select "other".
        
        ### The prompt to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        Topic Heirarchy: 
        {gen(max_tokens=5, stop=">", name='high_domain')} > {gen(max_tokens=5, stop=">", name='mid_domain')} > {gen(max_tokens=5, stop=[">", n], name='low_domain')}
        """
    return lm

model_paths = [
    "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP3-0.1-q8_0.gguf",
    "/data2/.shared_models/llama.cpp_models/Meta-Llama-3.1-70B-Instruct-IMP3-0.1-q8_0.gguf"
]

for model_path in model_paths:

    print(f"loading {model_path}")

    llm = models.LlamaCpp(
        model=model_path,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=4096
    )

    print("generating response...")

    result = llm + annotation_fn(prompt="How to build a website from scratch?")

    print(result)

    del llm