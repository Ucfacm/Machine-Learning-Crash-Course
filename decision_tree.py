import math
import pandas as pd

def entropy(p: float = None, s: pd.Series = None):
    if s:
        p = len(s[s == 1]) / len(s)
    return 0 if (p == 0 or p == 1) else -p * math.log(p, 2) - (1 - p) * math.log(1 - p, 2)

def gain(x, s_label, a: str):
    v = set(x[a])
    sub_sum = 0
    s = len(x[x[s_label] == 1]) / len(x)
    for i in v:
        subset = x[x[a] == i]
        sv_weight = len(subset) / len(x)
        s_v_p = len(subset[subset[s_label] == 1]) / len(subset)
        sub_sum += sv_weight * entropy(p=s_v_p)
    return entropy(p=s) - sub_sum, len(x[x[a] == 1]), len(x[x[a] == 0])

def ID3_layer(x, s_label):
    gain_a = 0
    pos_a = 0
    neg_a = 0
    attribute = None
    for a in x.drop(s_label, axis=1).columns:
        gain_b, pos_b, neg_b = gain(x=x, a=a, s_label=s_label)
        if gain_b >= gain_a:
            gain_a = gain_b
            pos_a = pos_b
            neg_a = neg_b
            attribute = a
    return attribute, gain_a, pos_a, neg_a

def ID3(x, s_label, depth, current_depth=0):
    if current_depth == depth:
        return
    attribute, gain_a, pos_a, neg_a = ID3_layer(x, s_label)
    print('depth:', current_depth, 'attribute:', attribute, 'gain:', gain_a, 'positive:', pos_a, 'negative:', neg_a)
    ID3(x[x[attribute] == 1].drop(attribute, axis=1), s_label, depth, current_depth + 1)
    ID3(x[x[attribute] == 0].drop(attribute, axis=1), s_label, depth, current_depth + 1)

def main():
    students_df = pd.read_csv('q_1/students.csv')
    ID3(x=students_df, s_label='A', depth=3)

if __name__ == '__main__':
    main()