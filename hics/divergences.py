# Created by Marcus Pappik

import numpy as np
import pandas as pd


def KLD(P: pd.DataFrame, Q: pd.DataFrame, homogenous=False):
    """Kullback-Leibler divergence
    P -- Conditional distribution
    Q -- Marginal distribution
    """
    if not homogenous:
        P = P.loc[P['value'].isin(Q['value']), ].reset_index(drop=True)
        Q = Q.loc[Q['value'].isin(P['value']), ].reset_index(drop=True)

    divergence_sum = 0
    deviations = {}
    for value, p_prob, q_prob in zip(P['value'], P['probability'], Q['probability']):
        divergence_sum += (p_prob * np.log2(p_prob / q_prob))
        deviations[value] = abs(q_prob - p_prob) / q_prob

    return (divergence_sum, deviations)


def JSD(P: pd.DataFrame, Q: pd.DataFrame):
    """Jensen-Shannon divergence
    """
    P = P.loc[P['value'].isin(Q['value']), ].reset_index(drop=True)
    Q = Q.loc[Q['value'].isin(P['value']), ].reset_index(drop=True)
    M = pd.DataFrame({'probability': (P.probability + Q.probability) * 0.5})
    return (KLD(P, M, homogenous=True) + KLD(Q, M, homogenous=True)) * 0.5


def KS(P: pd.DataFrame, Q: pd.DataFrame, should_normalize=False, max_divergence=1, min_divergence=0):
    """Kolmogorov–Smirnov test
    """
    current_divergence = 0

    cutpoints = np.random.rand(50) * (Q.iloc[-1]['value'] - Q.iloc[0]['value']) + Q.iloc[-1]['value']

    samples = Q['count'].sum()
    N = P['count'].sum()
    for cut in cutpoints:
        sample_probability = (Q['value'] <= cut).sum() / samples
        real_probability = (P['value'] <= cut).sum() / N
        current_divergence = max(current_divergence, abs(sample_probability - real_probability))
    if should_normalize:
        current_divergence = (current_divergence - min_divergence) / (max_divergence - min_divergence)
    return current_divergence
