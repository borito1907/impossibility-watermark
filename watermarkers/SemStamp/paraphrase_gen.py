from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer
import argparse
from tqdm import tqdm, trange
from datasets import load_from_disk, Dataset
from nltk import sent_tokenize
import torch
import pickle
from transformers import AutoTokenizer
# from dipper import DipperParaphraser
from .paraphrase_gen_utils import accept_by_bigram_overlap, accept_by_unigram_overlap, SParrot, query_openai, query_openai_bigram, gen_prompt, gen_bigram_prompt, extract_list   
from .sampling_utils import well_formed_sentence

device = 'cuda' if torch.cuda.is_available() else "cpu"
num_beams = 25

def pegasus_paraphrase(texts, 
                       tokenizer, 
                       paraphraser_name="tuner007/pegasus_paraphrase", 
                       device='cuda', 
                       num_beams = 10, 
                       temp=2, 
                       bigram=False, 
                       bert_threshold=0.03
                       ):
    paraphraser = PegasusForConditionalGeneration.from_pretrained(
        paraphraser_name).to(device)
    paraphraser_tokenizer = PegasusTokenizer.from_pretrained(paraphraser_name)

    def paraphrase(sents):
        '''
        Arguments:
            sents: list of sentences (max len under 60!)
        Returns:
            paraphrased: list of paraphrased sents
        '''
        batch = paraphraser_tokenizer(
            sents, truncation=True, padding='longest', return_tensors="pt", max_length=60).to(device)
        paraphrased_ids = paraphraser.generate(
            **batch, max_length=60, num_beams=num_beams, num_return_sequences=num_beams, temperature=temp, repetition_penalty=1.03)
        paraphrased = paraphraser_tokenizer.batch_decode(
            paraphrased_ids, skip_special_tokens=True)
        return paraphrased
    
    # dataset has to be a text list
    sents, data_len = [], []
    for text in tqdm(texts, desc="Tokenizer"):
        sent_list = sent_tokenize(text)
        sents.extend(sent_list)
        data_len.append(len(sent_list))
    paras = []
    
    for sent in tqdm(sents):
        paraphrased = paraphrase(sent)
        paraphrased = [well_formed_sentence(para) for para in paraphrased]
        if bigram:
            para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold)
        else: 
            para = paraphrased[0]
        paras.append(para)
    
    start_pos = 0
    output = []
    new_texts = []
    for l in data_len:
        output.append(paras[start_pos: start_pos+l])
        new_texts.append(sents[start_pos: start_pos+l])
        start_pos+=l
    new_dataset = Dataset.from_dict(
        {'text': new_texts, 'para_text': output})
    name = args.data_path + \
        f'-pegasus-bigram={bigram}-threshold={bert_threshold}'
    new_dataset.save_to_disk(name)
    return output

def parrot_paraphrase(parrot, texts, tokenizer, num_beams=10, bigram=False, save_to_disk=True, avg_sent_len=20, save_by_sents=False, bert_threshold=0.03, unigram = False):
    # modified parrot source code to have the num_beams argument
    def paraphrase(sent):
        para_phrases = parrot.augment(input_phrase=sent,
                                      use_gpu=True,
                                      diversity_ranker="levenshtein",
                                      do_diverse=True,
                                      max_return_phrases=num_beams,
                                      max_length=60,
                                      adequacy_threshold=0.8,
                                      fluency_threshold=0.8)
        return para_phrases

    sents, data_len = [], []
    for text in tqdm(texts, desc="Tokenizer"):
        sent_list = sent_tokenize(text)
        sents.extend(sent_list)
        data_len.append(len(sent_list))
    start_pos = 0
    paras = []
    total_paraphrased = []
    for sent in tqdm(sents):
        paraphrased = paraphrase(sent)
        paraphrased = [well_formed_sentence(
            para, end_sent=True) for para in paraphrased]
        total_paraphrased.append(paraphrased)
        if bigram:
            para = accept_by_bigram_overlap(sent, paraphrased, tokenizer, bert_threshold=bert_threshold)
        else:
            para = accept_by_unigram_overlap(sent, paraphrased, tokenizer, bert_threshold=bert_threshold)
        paras.append(para)
    start_pos = 0
    output = []
    new_texts = []
    if save_by_sents:
        for l in data_len:
            output.append(paras[start_pos: start_pos+l])
            new_texts.append(sents[start_pos: start_pos+l])
            start_pos += l
    elif save_to_disk:
        new_texts = texts
        for l in data_len:
            output.append(" ".join(paras[start_pos: start_pos+l]))
            start_pos += l
    new_dataset = Dataset.from_dict({'text': new_texts, 'para_text': output})   
    name = args.data_path + \
        f'-parrot-bigram={bigram}-threshold={bert_threshold}'
    new_dataset.save_to_disk(name)
    pkl_name = args.data_path + f'-parrot-bigram={bigram}-threshold={bert_threshold}-all_beams.pkl'
    with open(pkl_name, 'wb') as f:
        pickle.dump(total_paraphrased, f)
        f.close()
    return output

def paraphrase_openai(data_path, num_beams, bigram=False):
    dataset = load_from_disk(data_path)
    texts = dataset['text']
    new_texts = []
    all_paras = []
    for text in tqdm(texts, desc="Tokenizer"):
        sents = sent_tokenize(text)
        para_sents = [] # shape: len(sents) * len(paraphrases) (=10)
        fail = False # this fail applies to all sents in this text
    for i in range(len(sents)):
        if fail:
            continue # fast forward to next piece of text
        sent = sents[i]
        context = sents[:i]
        if bigram:
            para_ls = []
            num_iter = 0
            prompt = gen_bigram_prompt(sent, context, num_beams)
        # if insufficient number of para_sents generated, try again
        while(len(para_ls) < 5):
            para_str = query_openai_bigram(prompt)
            # use regex to extract list from string
            para_ls = extract_list(para_str)
            num_iter += 1
            # openai refuses to paraphrase, thendiscard
            if num_iter == 10:
                fail = True
            break
        if not fail:
            para_sents.append(para_ls)
        else:
            prompt = gen_prompt(sent, context)
            para = query_openai(prompt)
            para_sents.append(para)
    if not fail:
        new_texts.append(sents)
        all_paras.append(para_sents) 
    
    save_path = data_path + f'-openai-num_beams={num_beams}-bigram={bigram}'
    Dataset.from_dict({'text': new_texts, 'para_text': all_paras}).save_to_disk(save_path)
    return new_texts, all_paras

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--model_path', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--paraphraser', type=str,
                        default="pegasus", choices=['pegasus', 
                                                    'parrot', 
                                                    'openai',
                                                    'parrot-bigram', 
                                                    'pegasus-bigram', 
                                                    'openai-bigram'])
    parser.add_argument('--temp', type=float, default=2.0, help='decode temperature')
    parser.add_argument('--bert_threshold', type=float, default=0.0, help='threshold for bert similarity between original and paraphrased')
    parser.add_argument('--num_beams', type=int, default=25, help='number of beams for beam-search')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()

    dataset = load_from_disk(args.data_path)
    texts = dataset['text']

    if args.paraphraser == 'parrot':
        parrot = SParrot()
        parrot_paraphrase(parrot, texts, tokenizer, num_beams=args.num_beams, 
                            bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'parrot-bigram':
        parrot = SParrot()
        parrot_paraphrase(parrot, texts, tokenizer, bigram=True, num_beams=args.num_beams,
                            bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'pegasus-bigram':
        pegasus_paraphrase(texts, tokenizer, bigram=True, num_beams=args.num_beams, bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'pegasus':
        pegasus_paraphrase(texts, tokenizer, num_beams = args.num_beams, bert_threshold=args.bert_threshold)
    elif args.paraphraser == 'openai':
        new_texts, paras = paraphrase_openai(texts, args.num_beams, bigram=False)
    elif args.paraphraser == 'openai-bigram':
        new_texts, paras = paraphrase_openai(texts, args.num_beams, bigram=True)
    else:
        raise NotImplementedError
