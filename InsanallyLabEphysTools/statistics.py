import pingouin as pg
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

from .utility import rmnan

def removeNansForStats(data):
    for key in data.__dict__:
        data.__dict__[key] = rmnan(data.__dict__[key])
    return data

def nonparametricIndependentStatsAllToAll(data,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data = removeNansForStats(data)

    ### Create dataframe ###
    phase = len(data.preearly)*["pre-early"] + len(data.prelate)*["pre-late"] + len(data.preexpert)*["pre-expert"] \
            + len(data.postearly)*["post-early"] + len(data.postlate)*["post-late"] + len(data.postexpert)*["post-expert"]
    stage = len(data.preearly)*["pre"] + len(data.prelate)*["pre"] + len(data.preexpert)*["pre"] \
            + len(data.postearly)*["post"] + len(data.postlate)*["post"] + len(data.postexpert)*["post"]
    mod = (np.concatenate([data.preearly, data.prelate, data.preexpert, data.postearly, data.postlate, data.postexpert]))

    df = pd.DataFrame({"phase": phase, "stage": stage, "FR_mod": mod})



    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase'], detailed=True)
    print(anova)



    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between="phase", padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["p-corr"] <= 0.05][["A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data.__dict__:
        print(f"key {key} has {len(data.__dict__[key])} non-nan elements")
    return pairwise_mwu



def nonparametricIndependentStatsCompareToPreearly(data,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data = removeNansForStats(data)

    ### Create dataframe ###
    phase = len(data.preearly)*["pre-late"] + len(data.prelate)*["pre-late"] \
            + len(data.preearly)*["pre-expert"] + len(data.preexpert)*["pre-expert"] \
            + len(data.preearly)*["post-early"] + len(data.postearly)*["post-early"] \
            + len(data.preearly)*["post-late"] + len(data.postlate)*["post-late"] \
            + len(data.preearly)*["post-expert"] + len(data.postexpert)*["post-expert"]


    context = len(data.preearly)*["baseline"] + len(data.prelate)*["experimental"] \
            + len(data.preearly)*["baseline"] + len(data.preexpert)*["experimental"] \
            + len(data.preearly)*["baseline"] + len(data.postearly)*["experimental"] \
            + len(data.preearly)*["baseline"] + len(data.postlate)*["experimental"] \
            + len(data.preearly)*["baseline"] + len(data.postexpert)*["experimental"]

    mod = (np.concatenate([data.preearly, data.prelate,
            data.preearly, data.preexpert,
            data.preearly, data.postearly,
            data.preearly, data.postlate,
            data.preearly, data.postexpert]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})



    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()



    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data.__dict__:
        print(f"key {key} has {len(data.__dict__[key])} non-nan elements")
    return pairwise_mwu



def nonparametricIndependentStatsCompareToOwnEarly(data,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data = removeNansForStats(data)

    ### Create dataframe ###
    phase = len(data.preearly)*["pre-late"] + len(data.prelate)*["pre-late"] \
            + len(data.preearly)*["pre-expert"] + len(data.preexpert)*["pre-expert"] \
            + len(data.postearly)*["post-late"] + len(data.postlate)*["post-late"] \
            + len(data.postearly)*["post-expert"] + len(data.postexpert)*["post-expert"]


    context = len(data.preearly)*["baseline"] + len(data.prelate)*["experimental"] \
            + len(data.preearly)*["baseline"] + len(data.preexpert)*["experimental"] \
            + len(data.postearly)*["baseline"] + len(data.postlate)*["experimental"] \
            + len(data.postearly)*["baseline"] + len(data.postexpert)*["experimental"]

    mod = (np.concatenate([data.preearly, data.prelate,
            data.preearly, data.preexpert,
            data.postearly, data.postlate,
            data.postearly, data.postexpert]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})



    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()



    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data.__dict__:
        print(f"key {key} has {len(data.__dict__[key])} non-nan elements")
    return pairwise_mwu






def nonparametricIndependentStatsCompareToOther(data1,data2,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data1 = removeNansForStats(data1)
    data2 = removeNansForStats(data2)

    ### Create dataframe ###
    phase = len(data1.preearly)*["pre-early"] + len(data2.preearly)*["pre-early"] \
            + len(data1.prelate)*["pre-late"] + len(data2.prelate)*["pre-late"] \
            + len(data1.preexpert)*["pre-expert"] + len(data2.preexpert)*["pre-expert"] \
            + len(data1.postearly)*["post-early"] + len(data2.postearly)*["post-early"] \
            + len(data1.postlate)*["post-late"] + len(data2.postlate)*["post-late"] \
            + len(data1.postexpert)*["post-expert"] + len(data2.postexpert)*["post-expert"]


    context = len(data1.preearly)*["A"] + len(data2.preearly)*["B"] \
            + len(data1.prelate)*["A"] + len(data2.prelate)*["B"] \
            + len(data1.preexpert)*["A"] + len(data2.preexpert)*["B"] \
            + len(data1.postearly)*["A"] + len(data2.postearly)*["B"] \
            + len(data1.postlate)*["A"] + len(data2.postlate)*["B"] \
            + len(data1.postexpert)*["A"] + len(data2.postexpert)*["B"]

    mod = (np.concatenate([data1.preearly, data2.preearly,
            data1.prelate, data2.prelate,
            data1.preexpert, data2.preexpert,
            data1.postearly, data2.postearly,
            data1.postlate, data2.postlate,
            data1.postexpert, data2.postexpert]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})

    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()

    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data1.__dict__:
        print(f"key {key} has {len(data1.__dict__[key])} and {len(data2.__dict__[key])} non-nan elements")
    return pairwise_mwu


def nonparametricIndependentStatsCompareToOtherSmall(data1,data2,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data1 = removeNansForStats(data1)
    data2 = removeNansForStats(data2)

    ### Create dataframe ###
    phase = len(data1.early)*["early"] + len(data2.early)*["early"] \
            + len(data1.late)*["late"] + len(data2.late)*["late"] \
            + len(data1.expert)*["expert"] + len(data2.expert)*["expert"] \

    context = len(data1.early)*["A"] + len(data2.early)*["B"] \
            + len(data1.late)*["A"] + len(data2.late)*["B"] \
            + len(data1.expert)*["A"] + len(data2.expert)*["B"] \

    mod = (np.concatenate([data1.early, data2.early,
            data1.late, data2.late,
            data1.expert, data2.expert,]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})

    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()

    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data1.__dict__:
        print(f"key {key} has {len(data1.__dict__[key])} and {len(data2.__dict__[key])} non-nan elements")
    return pairwise_mwu







def nonparametricIndependentStatsCompareToPrevious(data,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data = removeNansForStats(data)

    ### Create dataframe ###
    phase = len(data.preearly)*["pre-late"] + len(data.prelate)*["pre-late"] \
            + len(data.prelate)*["pre-expert"] + len(data.preexpert)*["pre-expert"] \
            + len(data.postearly)*["post-late"] + len(data.postlate)*["post-late"] \
            + len(data.postlate)*["post-expert"] + len(data.postexpert)*["post-expert"]


    context = len(data.preearly)*["baseline"] + len(data.prelate)*["experimental"] \
            + len(data.prelate)*["baseline"] + len(data.preexpert)*["experimental"] \
            + len(data.postearly)*["baseline"] + len(data.postlate)*["experimental"] \
            + len(data.postlate)*["baseline"] + len(data.postexpert)*["experimental"]

    mod = (np.concatenate([data.preearly, data.prelate,
            data.prelate, data.preexpert,
            data.postearly, data.postlate,
            data.postlate, data.postexpert]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})



    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()



    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data.__dict__:
        print(f"key {key} has {len(data.__dict__[key])} non-nan elements")
    return pairwise_mwu



def nonparametricIndependentStatsCompareToPreviousSmall(data,parametric=False):
    '''
    Data should be a SimpleNamespace with six fields within it labeled ['preearly','prelate','preexpert','postearly','postlate','postexpert']
    '''
    data = removeNansForStats(data)

    ### Create dataframe ###
    phase = len(data.early)*["pre-late"] + len(data.late)*["pre-late"] \
            + len(data.late)*["pre-expert"] + len(data.expert)*["pre-expert"] \

    context = len(data.early)*["baseline"] + len(data.late)*["experimental"] \
            + len(data.late)*["baseline"] + len(data.expert)*["experimental"] \

    mod = (np.concatenate([data.early, data.late,
            data.late, data.expert]))

    df = pd.DataFrame({"phase": phase, "context": context, "FR_mod": mod})

    ### Anova ###
    anova = pg.anova(data=df, dv='FR_mod', between=['phase','context'], detailed=True)
    print(anova)
    print()

    ### Create and print results ###
    pairwise_mwu = pg.pairwise_tests(data=df, dv='FR_mod', between=['phase','context'], padjust="fdr_bh", parametric=parametric, )
    print(pairwise_mwu[pairwise_mwu["Contrast"] == "phase * context"][["phase","A", "B", "p-unc", "p-corr", "hedges"]])
    for key in data.__dict__:
        print(f"key {key} has {len(data.__dict__[key])} non-nan elements")
    return pairwise_mwu


def mannwhitneycompareall(data):
	data = removeNansForStats(data)

	p1 = mannwhitneyu(data.preearly,data.prelate).pvalue
	p2 = mannwhitneyu(data.prelate,data.preexpert).pvalue
	p3 = mannwhitneyu(data.preearly,data.preexpert).pvalue

	p4 = mannwhitneyu(data.postearly,data.postlate).pvalue
	p5 = mannwhitneyu(data.postlate,data.postexpert).pvalue
	p6 = mannwhitneyu(data.postearly,data.postexpert).pvalue

	pvals = [p1,p2,p3,p4,p5,p6]
	_,pvals_corr = pg.multicomp(pvals,method='fdr_bh')

	print(f"pre-early   n = {len(data.preearly)}")
	print(f"pre-late    n = {len(data.prelate)}")
	print(f"pre-expert  n = {len(data.preexpert)}")
	print(f"post-early  n = {len(data.postearly)}")
	print(f"post-late   n = {len(data.postlate)}")
	print(f"post-expert n = {len(data.postexpert)}")
	print(f"pre-early  to pre-late   : {pvals_corr[0]}")
	print(f"pre-late   to pre-expert : {pvals_corr[1]}")
	print(f"pre-early  to pre-expert : {pvals_corr[2]}")
	print(f"post-early to post-late  : {pvals_corr[3]}")
	print(f"post-late  to post-expert: {pvals_corr[4]}")
	print(f"post-early to post-expert: {pvals_corr[5]}")

def mannwhitneycomparesmall(data):
	data = removeNansForStats(data)

	p1 = mannwhitneyu(data.early,data.late).pvalue
	p2 = mannwhitneyu(data.late,data.expert).pvalue
	p3 = mannwhitneyu(data.early,data.expert).pvalue

	pvals = [p1,p2,p3]
	_,pvals_corr = pg.multicomp(pvals,method='fdr_bh')

	print(f"early   n = {len(data.early)}")
	print(f"late    n = {len(data.late)}")
	print(f"expert  n = {len(data.expert)}")
	print(f"early  to late   : {pvals_corr[0]}")
	print(f"late   to expert : {pvals_corr[1]}")
	print(f"early  to expert : {pvals_corr[2]}")

def ttestindcompareall(data):
	data = removeNansForStats(data)

	p1 = ttest_ind(data.preearly,data.prelate).pvalue
	p2 = ttest_ind(data.prelate,data.preexpert).pvalue
	p3 = ttest_ind(data.preearly,data.preexpert).pvalue

	p4 = ttest_ind(data.postearly,data.postlate).pvalue
	p5 = ttest_ind(data.postlate,data.postexpert).pvalue
	p6 = ttest_ind(data.postearly,data.postexpert).pvalue

	pvals = [p1,p2,p3,p4,p5,p6]
	_,pvals_corr = pg.multicomp(pvals,method='fdr_bh')

	print(f"pre-early   n = {len(data.preearly)}")
	print(f"pre-late    n = {len(data.prelate)}")
	print(f"pre-expert  n = {len(data.preexpert)}")
	print(f"post-early  n = {len(data.postearly)}")
	print(f"post-late   n = {len(data.postlate)}")
	print(f"post-expert n = {len(data.postexpert)}")
	print(f"pre-early  to pre-late   : {pvals_corr[0]}")
	print(f"pre-late   to pre-expert : {pvals_corr[1]}")
	print(f"pre-early  to pre-expert : {pvals_corr[2]}")
	print(f"post-early to post-late  : {pvals_corr[3]}")
	print(f"post-late  to post-expert: {pvals_corr[4]}")
	print(f"post-early to post-expert: {pvals_corr[5]}")