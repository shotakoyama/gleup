import dataclasses
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from functools import cache
from multiprocessing import Pool
from pathlib import Path


@dataclasses.dataclass
class Accumulator:
    match: int = 0
    penal: int = 0
    denom: int = 0

    def __add__(self, other):
        match = self.match + other.match
        penal = self.penal + other.penal
        denom = self.denom + other.denom
        return type(self)(match, penal, denom)

    @classmethod
    @cache
    def make(cls, s, r, c):
        match = min(r, c)
        penal = max(0, min(s, c) - r)
        return cls(match, penal, c)

    def p(self):
        if self.denom == 0:
            return 1.0
        numer = self.match - self.penal
        assert numer >= 0, self # The numerator must be greater than or equal to 0.
        return numer / self.denom

    def rectify(self):
        penal = min(self.match, self.penal)
        return type(self)(self.match, penal, self.denom)


@dataclasses.dataclass
class NgramStat:
    src_dict: dict
    ref_dict: dict
    cor_dict: dict

    def __getitem__(self, key):
        s = self.src_dict.get(key, 0)
        r = self.ref_dict.get(key, 0)
        c = self.cor_dict.get(key, 0)
        return s, r, c

    def accumlate(self):
        lst = [
            Accumulator.make(*self[key])
            for key
            in set(self.cor_dict)]
        return sum(lst, start = Accumulator())


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-s', dest = 'src_path')
    parser.add_argument('-r', nargs = '+', dest = 'ref_path_list')
    parser.add_argument('-c', nargs = '+', dest = 'cor_path_list')
    parser.add_argument('-i', type = int, default = 500, dest = 'num_iter')
    parser.add_argument('-n', type = int, default = 4)
    parser.add_argument('-d', type = int, default = 2, dest = 'digit')
    parser.add_argument('-p', type = int, default = 8, dest = 'num_proc')
    parser.add_argument('-f', action = 'store_true', dest = 'fix_seed')
    return parser.parse_args()


def main(args):
    dr_rlen, dc_clen, cdrn_accum = load(args)
    id_rindex = make_id_rindex(args.num_iter, len(dr_rlen), len(dr_rlen[0]), args.fix_seed)
    for c, drn_accum in enumerate(cdrn_accum):
        name = Path(args.cor_path_list[c]).name
        with Pool(args.num_proc) as pool:
            gleus = pool.starmap(calc_gleu, [(drn_accum, d_rindex, dr_rlen, dc_clen[c]) for d_rindex in id_rindex])
        gleu = np.mean(gleus)
        print(name + '\t' + round_half_up(100 * gleu, args.digit))


def load(args):
    d_src = load_text(args.src_path)
    dr_ref = list(zip(*[load_text(path) for path in args.ref_path_list]))
    cd_cor = [load_text(path) for path in args.cor_path_list]
    assert len(d_src) == len(dr_ref)
    assert all(len(d_src) == len(d_cor) for d_cor in cd_cor)
    dr_rlen = np.array([[len(split_ngram(1, r)) for r in r_ref] for r_ref in dr_ref], dtype = int)
    cd_clen = np.array([[len(split_ngram(1, c)) for c in d_cor] for d_cor in cd_cor], dtype = int)
    accum = aggreg(args.n, d_src, dr_ref, cd_cor)
    return dr_rlen, cd_clen, accum


def load_text(path):
    with open(path) as f:
        data = [x.rstrip('\n') for x in f]
    return data


@cache
def split_ngram(n, sent): # word n-gram
    sent = sent.split()
    return [' '.join(sent[i : i + n]) for i in range(len(sent) - n + 1)]


@cache
def ngram_counter(n, sent):
    return dict(Counter(split_ngram(n, sent)))


def aggreg(max_n, d_src, dr_ref, cd_cor):
    return [[[
        make_n_accum(max_n, s, r, c)
        for r in r_ref]
        for s, r_ref, c in zip(d_src, dr_ref, d_cor)]
        for d_cor in cd_cor]


@cache
def make_n_accum(max_n, s, r, c):
    return [make_accum(n, s, r, c) for n in range(1, max_n + 1)]


def make_accum(n, s, r, c):
    s_cnt = ngram_counter(n, s)
    r_cnt = ngram_counter(n, r)
    c_cnt = ngram_counter(n, c)
    accum = NgramStat(s_cnt, r_cnt, c_cnt).accumlate()
    return accum


def make_id_rindex(num_iter, d, r, fix = False):
    if fix:
        np.random.seed(0)
    return np.random.randint(r, size = (num_iter, d))


def calc_gleu(drn_accum, d_rindex, dr_rlen, d_clen):
    dn_accum = [rn_accum[rindex] for rn_accum, rindex in zip(drn_accum, d_rindex)]
    nd_accum = list(zip(*dn_accum))
    n_accum = [sum(d_accum, start = Accumulator()) for d_accum in nd_accum]
    ps = [accum.rectify().p() for accum in n_accum] # rectify() is called to make numer >= 0.
    with np.errstate(divide = 'ignore'):
        logps = np.log(ps)
    loggmeanp = logps.mean()
    rlen, clen = sum([dr_rlen[d, r] for d, r in enumerate(d_rindex)]), sum(d_clen)
    logbp = log_brevity_penalty(rlen, clen)
    return np.exp(logbp + loggmeanp)


@cache
def log_brevity_penalty(rlen, clen):
    if (rlen, clen) == (0, 0) or rlen < clen:
        return 0.0
    elif clen == 0:
        return -np.inf
    else:
        return 1.0 - rlen / clen


def round_half_up(x, digit):
    if x == np.inf or x == -np.inf:
        return str(x)
    digit = Decimal('0.' + '0' * digit)
    x = Decimal(str(x)).quantize(digit, rounding = ROUND_HALF_UP)
    return str(x)


if __name__ == '__main__':
    args = parse_args()
    main(args)

