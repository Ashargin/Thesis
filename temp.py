---
tags:
- dna_bert
---
```
NUM_CLASSES = number of the classes in your data

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
tokenizer = AutoTokenizer.from_pretrained(
   zhihan1996/DNA_bert_6, do_lower_case=False
)

model = AutoModelForSequenceClassification.from_pretrained(
     zhihan1996/DNA_bert_6, num_labels=NUM_CLASSES
)

def return_kmer(seq, K=6):
    """
    This function outputs the K-mers of a sequence
    Parameters
    ----------
    seq : str
        A single sequence to be split into K-mers
    K : int, optional
        The length of the K-mers, by default 6
    Returns
    -------
    kmer_seq : str
        A string of K-mers separated by spaces
    """

    kmer_list = []
    for x in range(len(seq) - K + 1):
        kmer_list.append(seq[x : x + K])

    kmer_seq = " ".join(kmer_list)
    return kmer_seq

sequence = your DNA sequences

train_kmers = [return_kmer(seq) for seq in sequence]

train_encodings = tokenizer.batch_encode_plus(
    train_kmers,
    max_length=512,  # max len of BERT
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
```