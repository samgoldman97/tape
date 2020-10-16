""" Create evotune train val test of the file 

python scripts/evosplit.py --fasta-in data/olea_finetune_fullseq.fasta --ref-fasta data/ref_seq.fasta --outdir data

"""


import argparse
import os
from itertools import groupby
import typing
from typing import Iterable, Tuple
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
import numpy as np 
import pdb

def fasta_iter(fasta_name: str) -> Iterable[Tuple[str, str]]:
    """fasta_iter.

     modified from Brent Pedersen
     Correct Way To Parse A Fasta File In Python
     given a fasta file. yield tuples of header, sequence
     
    Args:
        fasta_name (str): fasta_name

    Returns: 
        Iterable[tuple] of (headerStr, seq)
    """
    # open fasta file
    fh = open(fasta_name)

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)

def write_fasta(seq_dict: dict, fasta_name: str) -> None:
    """write_fasta.

     modified from Brent Pedersen
     Correct Way To Parse A Fasta File In Python
     given a fasta file. yield tuples of header, sequence
     
    Args:
        seq_dict (dict): name to sequence map
        fasta_name (str): fasta_name

    Returns: 
    """
    # open fasta file
    fh = open(fasta_name, "w")
    for k, v in seq_dict.items():
        fh.write(f">{k.strip()}\n{v.strip().upper()}\n")
    fh.close()


def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta-in", help="name of fasta input")
    parser.add_argument("--outdir", help="name of fasta out")
    parser.add_argument("--ref-fasta", help="File containing the ref seq")
    return parser.parse_args()



def evo_split(args): 
    """ Use split """
    ref_seq = list(fasta_iter(args.ref_fasta))[0][1]
    os.makedirs(args.outdir, exist_ok=True)

    fasta_seq_to_name = {seq: name_ for name_, seq  in fasta_iter(args.fasta_in)}
    fasta_seqs = list(fasta_seq_to_name.keys())#[:1000]

    edit_distances = {}
    probabilities = {}

    filtered_ct = 0 
    for seq in tqdm(fasta_seqs): 
        edit_distances[seq] =  levenshtein_distance(seq, ref_seq)
        if edit_distances[seq] < 400: 
            probabilities[seq] = np.power(edit_distances[seq], 4)
        else: 
            filtered_ct += 1
            print("FILTERED")
    print(f"Number filtered: {filtered_ct}")

    # Normalize these
    total_mass = np.sum(list(probabilities.values()))
    new_probs = dict()
    # Hold sequence and sequnence probabilities
    seqs, probs= [], []
    for k,v in probabilities.items(): 
        seqs.append(k)
        probs.append(v/total_mass)


    # Hold out 10% of this according to sampling with levenshtein distance;
    # higher dist is more likely sampled

    sample_size = int(0.1 * len(seqs))

    selected_seqs = np.random.choice(seqs, size=sample_size, 
                                     replace=False, p = probs)

    orig_set = set(fasta_seqs)
    val_test_set = set(selected_seqs)
    train_set = orig_set.difference(val_test_set) 

    # Split val_test further 
    np.random.shuffle(selected_seqs)
    split_loc = int(len(selected_seqs) / 2)
    val = selected_seqs[:split_loc]
    test = selected_seqs[split_loc:]


    # Make dicts! 
    train_dict = {fasta_seq_to_name[i]: i  for i in train_set} 
    val_dict = {fasta_seq_to_name[i]: i  for i in val} 
    test_dict = {fasta_seq_to_name[i]: i   for i in test} 



    # Now write these all out
    write_fasta(train_dict, 
                os.path.join(args.outdir, f"finetune_train.fasta"))
    write_fasta(val_dict, 
                os.path.join(args.outdir, f"finetune_valid.fasta"))
    write_fasta(test_dict, 
                os.path.join(args.outdir, f"finetune_test.fasta"))

if __name__=="__main__": 
    args = get_args()
    evo_split(args)



