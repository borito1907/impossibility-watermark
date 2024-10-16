'''
produce model generation
'''
import pprint
import argparse
import os
import sys
from datasets import load_from_disk, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from .sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from .sampling_utils import extract_prompt_from_text
from .sampling_lsh_utils import lsh_reject_completion
# TODO: This is probably from k-SemStamp. It generates a bug right now.
from sampling_kmeans_utils import embed_gen_list, get_cluster_centers, kmeans_reject_completion, load_embeds

nltk.download('punkt')

PUNCTS = '.,!?'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', type=str, help='path to huggingface dataset that has a column "text"')
    parser.add_argument(
        '--model', type=str, help='str model name to generate continuation. huggingface/openai', default="facebook/opt-1.3b")
    parser.add_argument(
        '--embedder', default="",type=str, help='str model name to embed sentences')
    parser.add_argument('--len_prompt', '-l', default=32,
                        help='MAX length of prompt')
    parser.add_argument('--max_new_tokens', type=int, default=205)
    parser.add_argument('--min_new_tokens', type=int, default=195)
    parser.add_argument('--lmbd', type=float, default=0.25,
                        help='ratio of valid sentences')
    parser.add_argument('--delta', type=float, default=0,
                        help='logit augmentation for baseline or margin size for lsh and kmeans')
    parser.add_argument('--sp_mode', type=str,
                        choices=['lsh', 'kmeans'])
    parser.add_argument('--sp_dim', type=int, default=3, help='number of partitions in the embedding space. default 3 for semstamp and 8 for k-semstamp')
    parser.add_argument('--embed_path', type=str,
                        help='path to precomputed embed for training kmeans')
    parser.add_argument('--cc_path', type=str,
                        help='kmeans precomputed cluster centers data')
    parser.add_argument('--train_data', type=str,
                        help="train_data for kmeans clusters")
    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    # NOTE: currently, no batching
    is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get(
        'TRANSFORMERS_OFFLINE') == '1'

    dataset = load_from_disk(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=is_offline)
    folder_name = os.path.join(args.data, args.embedder)
    # block \n
    bad_words_ids = tokenizer(
        "\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device='cuda').tolist()
    
    # TODO: They didn't specify the device in their code.
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Removed repetition penalty argument from argparse, since they didn't have it.
    gen_config = GenerationConfig.from_pretrained(
        args.model,
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=0,
        repetition_penalty=1.1,
        bad_words_ids=bad_words_ids,
        # top_p=0.96,
        local_files_only=is_offline
    )


    name = os.path.join(
        folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-rep_p={1.1}")
        # folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-seed={args.seed_scheme}-rep_p={args.rep_p}")
    
    
    
    name = os.path.join(
        folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-rep_p={1.1}")
        # folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-seed={args.seed_scheme}-rep_p={args.rep_p}")
    
    
    if args.sp_mode == "lsh":
        # TODO: Fix lsh_model_path
        lsh_model = SBERTLSHModel(lsh_model_path=None,
                                  device=args.device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base')
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=is_offline).to(args.device)
        model.eval()
        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            response = lsh_reject_completion(
                prompt,
                model, tokenizer, gen_config,
                lsh_model, args.sp_dim,
                lmbd=args.lmbd,
                device=args.device,
                margin=args.delta)
            
            
            # TODO: This returns a tuple.
            print(prompt)
            print(response)

            ex['generated_text'] = response[0].strip()
            return ex

    # Create new dataset
    temp_dataset = dataset.map(text_to_generated_text, batch_size=1)
    print(f"temp dataset: {temp_dataset}")

    sample_folder_name = os.path.join(
        folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-rep_p={1.1}")
        # TODO: This was the old version. folder_name, f"lmbd={args.lmbd}-{args.sp_mode}-{args.delta}-{args.sp_dim}-len={args.min_new_tokens}-{args.max_new_tokens}-seed={args.seed_scheme}-rep_p={args.rep_p}")    
    os.makedirs(sample_folder_name, exist_ok=True)

    original_texts = temp_dataset['text']
    generated_texts = temp_dataset['generated_text']
    print(f"original texts: {original_texts}")
    print(f"generated texts: {generated_texts}")

    num_new_sentences = np.sum([len(sent_tokenize(t)) for t in generated_texts])
    print(f"Number of new sentences: {num_new_sentences}")
    

    # Save to CSV
    df = pd.DataFrame({'original_text': original_texts, 'generated_text': generated_texts})
    df.to_csv(f"{sample_folder_name}/results.csv", index=False)