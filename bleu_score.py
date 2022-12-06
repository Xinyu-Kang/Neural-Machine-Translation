
'''Calculate BLEU score for one reference and one hypothesis'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Get all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of token ids or words representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    if len(seq) >= n:
        return [seq[i:i+n] for i in range(len(seq)-n+1)]
    else:
        return []


def n_gram_precision(reference, candidate, n):
    '''Compute the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    reference_n_gram = grouper(reference, n)
    candidate_n_gram = grouper(candidate, n)
    len_c = len(candidate_n_gram)
    if len_c == 0:
        return 0
    match_c = 0
    for c in candidate_n_gram:
        if c in reference_n_gram:
            match_c += 1
    return match_c / len_c


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    len_r = len(reference)
    len_c = len(candidate)
    if len_c == 0:
        return 0
    breavity = len_r / len_c
    if breavity < 1:
        return 1
    else:
        return exp(1-breavity)


def BLEU_score(reference, candidate, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''
    bp = brevity_penalty(reference, candidate)
    p_product = 1
    for i in range(n+1):
        p = n_gram_precision(reference, candidate, i)
        p_product *= p
    score = bp * (p_product ** (1/n))
    return score
