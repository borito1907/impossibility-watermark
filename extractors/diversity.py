from textdiversity import (
    TokenSemantics, DocumentSemantics, AMR, # semantics
    DependencyParse, ConstituencyParse,     # syntactical
    PartOfSpeechSequence,                   # morphological
    Rhythmic                                # phonological
)
from lexical_diversity import lex_div as ld
from nltk import ngrams
from nltk.tokenize import word_tokenize
import numpy as np


class DocumentSemanticDiversity:
    def __init__(self):
        self.metric = DocumentSemantics({"normalize": False})
    
    def evaluate(self, dataset):
        return dataset, self.metric(dataset['text'])
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Anything lower than 1 means that the changes made 
        to the text reduced diversity. 
        """
        before_dataset, before_div = self.evaluate(before_dataset)
        after_dataset, after_div   = self.evaluate(after_dataset)
        div = np.nan_to_num(after_div / before_div)
        return after_dataset, div
    
class DocumentDependencyParseDiversity:
    def __init__(self):
        self.metric = DependencyParse({"normalize": False})
    
    def evaluate(self, dataset):
        return dataset, self.metric(dataset['text'])
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Anything lower than 1 means that the changes made 
        to the text reduced diversity. 
        """
        before_dataset, before_div = self.evaluate(before_dataset)
        after_dataset, after_div   = self.evaluate(after_dataset)
        div = np.nan_to_num(after_div / before_div)
        return after_dataset, div

class DocumentPartOfSpeechSequenceDiversity:
    def __init__(self):
        self.metric = PartOfSpeechSequence({"normalize": False})
    
    def evaluate(self, dataset):
        return dataset, self.metric(dataset['text'])
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Anything lower than 1 means that the changes made 
        to the text reduced diversity. 
        """
        before_dataset, before_div = self.evaluate(before_dataset)
        after_dataset, after_div   = self.evaluate(after_dataset)
        div = np.nan_to_num(after_div / before_div)
        return after_dataset, div    

class MATTRDiversity:
    def __init__(self):
        self.metric = LDHelper().mattr
    
    def evaluate(self, dataset):
        return dataset, self.metric(dataset['text'])
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Anything lower than 1 means that the changes made 
        to the text reduced diversity. 
        """
        before_dataset, before_div = self.evaluate(before_dataset)
        after_dataset, after_div   = self.evaluate(after_dataset)
        div = np.nan_to_num(after_div / before_div)
        return after_dataset, div   
    
class UniqueBigramsDiversity:
    def __init__(self):
        self.metric = UniqueNgramHelper().bigrams
    
    def evaluate(self, dataset):
        return dataset, self.metric(dataset['text'])
    
    def evaluate_before_and_after(self, before_dataset, after_dataset, annotate_after_dataset=True):
        """
        Anything lower than 1 means that the changes made 
        to the text reduced diversity. 
        """
        before_dataset, before_div = self.evaluate(before_dataset)
        after_dataset, after_div   = self.evaluate(after_dataset)
        div = np.nan_to_num(after_div / before_div)
        return after_dataset, div   

class LDHelper:

    def _flemmatize(self, corpus):
        flemmas = []
        for doc in corpus:
            flemmas.extend(ld.flemmatize(doc))
        return flemmas

    def ttr(self, coprus):
        return ld.ttr(self._flemmatize(coprus))

    def root_ttr(self, coprus):
        return ld.root_ttr(self._flemmatize(coprus))

    def log_ttr(self, coprus):
        return ld.log_ttr(self._flemmatize(coprus))

    def maas_ttr(self, coprus):
        return ld.maas_ttr(self._flemmatize(coprus))

    def msttr(self, coprus):
        return ld.msttr(self._flemmatize(coprus))

    def mattr(self, coprus):
        return ld.mattr(self._flemmatize(coprus))

    def hdd(self, coprus):
        return ld.hdd(self._flemmatize(coprus))

    def mtld(self, coprus):
        return ld.mtld(self._flemmatize(coprus))

    def mtld_ma_wrap(self, coprus):
        return ld.mtld_ma_wrap(self._flemmatize(coprus))

    def mtld_ma_bid(self, coprus):
        return ld.mtld_ma_bid(self._flemmatize(coprus))


class UniqueNgramHelper:

    def _tokenize(self, corpus):
        tokens = []
        for doc in corpus:
            tokens.extend(word_tokenize(doc))
        return tokens

    def _make_unique(self, n_gram_generator):
        return len(set(list(n_gram_generator)))

    def unigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 1)
        return self._make_unique(n_gram_generator)

    def bigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 2)
        return self._make_unique(n_gram_generator)

    def trigrams(self, corpus):
        tokens = self._tokenize(corpus)
        n_gram_generator = ngrams(tokens, 3)
        return self._make_unique(n_gram_generator)
    
if __name__ == "__main__":
    
    from fada.transform import Transform
    from fada.augmenter import Augmenter
    from datasets import load_dataset
    import sibyl
    import numpy as np

    dataset_config = ("glue", "sst2")
    task_name = "sentiment"

    dataset = load_dataset(*dataset_config, split="train[:3]")
    dataset = dataset.rename_column("sentence", "text")

    transforms = [
        sibyl.ChangeHypernym,
        sibyl.ChangeHyponym,
        sibyl.InsertPunctuationMarks
    ]
    transforms = [Transform(t, task_name=task_name) for t in transforms]

    num_augmentations_per_record = 1
    num_transforms_to_apply = 1
    batch_size = 1

    # uniform sampling probabilities
    uni_augmenter = Augmenter(
        dataset=dataset, 
        transforms=transforms,  
        transform_probabilities=None,
        num_augmentations_per_record=num_augmentations_per_record,
        num_transforms_to_apply=num_transforms_to_apply,
        keep_originals=False,
        batch_size=batch_size)
    aug_dataset = uni_augmenter.augment()

    metrics = [
        DocumentSemanticDiversity(),
        DocumentDependencyParseDiversity(),
        DocumentPartOfSpeechSequenceDiversity(),
        MATTRDiversity(),
        UniqueBigramsDiversity()
    ]

    print(f"original_dataset_text: {dataset['text']}")
    print(f"augmented_dataset_text: {aug_dataset['text']}")

    for metric in metrics:
        metric_name = metric.__class__.__name__
        print(f"Calculating {metric_name}...")

        dataset, orig_div = metric.evaluate(dataset)
        aug_dataset, aug_div = metric.evaluate(aug_dataset)
        aug_dataset, diff_div = metric.evaluate_before_and_after(dataset, aug_dataset)

        print(f"original_{metric_name}_score: {orig_div}")
        print(f"augmented_{metric_name}_score: {aug_div}")
        print(f"diffed_{metric_name}_scores: {diff_div}")

    # (fada) C:\Users\fabri\Documents\GitHub\fada>python -m fada.extractors.diversity      
    # original_dataset_text: ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ']
    # augmented_dataset_text: ['wrap new humour from the parental construct ', 'repress no substance , only labored humour ', 'that loves its characters and ? communicates something rather ; beautiful about human nature ? ']
    # Calculating DocumentSemanticDiversity...
    # original_DocumentSemanticDiversity_score: 1.7467101260393558
    # augmented_DocumentSemanticDiversity_score: 1.4460646222521027
    # diffed_DocumentSemanticDiversity_scores: 0.8278789941700497
    # Calculating DocumentDependencyParseDiversity...
    # original_DocumentDependencyParseDiversity_score: 2.9663878358148343
    # augmented_DocumentDependencyParseDiversity_score: 2.9478208621973376
    # diffed_DocumentDependencyParseDiversity_scores: 0.9937408812855395
    # Calculating DocumentPartOfSpeechSequenceDiversity...
    # original_DocumentPartOfSpeechSequenceDiversity_score: 1.6879078928779354
    # augmented_DocumentPartOfSpeechSequenceDiversity_score: 1.8470650772229111
    # diffed_DocumentPartOfSpeechSequenceDiversity_scores: 1.0942925766367546
    # Calculating MATTRDiversity...
    # original_MATTRDiversity_score: 0.9285714285714286
    # augmented_MATTRDiversity_score: 0.8928571428571429
    # diffed_MATTRDiversity_scores: 0.9615384615384616
    # Calculating UniqueBigramsDiversity...
    # original_UniqueBigramsDiversity_score: 25
    # augmented_UniqueBigramsDiversity_score: 28
    # diffed_UniqueBigramsDiversity_scores: 1.12