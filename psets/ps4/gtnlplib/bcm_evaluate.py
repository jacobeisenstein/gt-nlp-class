'''
Based on https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
'''
import numpy as np
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / p_den
    r = 0 if r_den == 0 else r_num / r_den
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, document):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(document.clusters, document.gold)
        else:
            pn, pd = self.metric(document.clusters, document.mention_to_gold)
            rn, rd = self.metric(document.gold, document.mention_to_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / self.r_den

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / self.p_den


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[mention_to_gold[m]] += 1
        for c2, count in gold_counts.items():
            correct += count * count

        num += correct / len(c)
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    gold_clusters = [c for c in gold_clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)

    
def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))
    