#!/usr/bin/env python3
from __future__ import annotations
import math
from typing import Iterable
from scipy import stats as sp_stats

def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values)/len(values)

def stdev(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2: return 0.0
    avg=mean(values)
    return math.sqrt(sum((v-avg)**2 for v in values)/(len(values)-1))

def ci95(values: Iterable[float]):
    values = list(values)
    if len(values) < 2: return mean(values), mean(values)
    avg=mean(values)
    se=stdev(values)/math.sqrt(len(values))
    return avg-1.96*se, avg+1.96*se

def ttest_paired(a,b):
    a,b=list(a),list(b)
    if len(a)!=len(b) or len(a)<2: return 0.0,1.0
    t,p=sp_stats.ttest_rel(a,b)
    return float(t),float(p)

def cohen_d(a,b):
    a,b=list(a),list(b)
    pooled=math.sqrt((stdev(a)**2+stdev(b)**2)/2)
    if pooled==0: return 0.0
    return (mean(a)-mean(b))/pooled
