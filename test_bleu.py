from bleu_score import *

if __name__ == "__main__":
    print("Testing Bleu ...\n")
    reference = '''\
it is a guide to action that ensures that the military will always heed
party commands'''.strip().split()
    candidate = '''\
it is a guide to action which ensures that the military always obeys the
commands of the party'''.strip().split()
    print("Reference: {}".format(reference))
    print("Candidate: {}".format(candidate))
    print("\n")

    reference_unigram = grouper(reference, 1)
    candidate_unigram = grouper(candidate, 1)
    reference_bigram = grouper(reference, 2)
    candidate_bigram = grouper(candidate, 2)
    print("Reference Unigram: {}".format(reference_unigram))
    print("Candidate Unigram: {}".format(candidate_unigram))
    print("Reference Bigram: {}".format(reference_bigram))
    print("Candidate Bigram: {}".format(candidate_bigram))
    print("\n")

    print("Unigram:")
    unigram_precision = n_gram_precision(reference, candidate, 1)
    print("Bigram:")
    bigram_precision = n_gram_precision(reference, candidate, 2)
