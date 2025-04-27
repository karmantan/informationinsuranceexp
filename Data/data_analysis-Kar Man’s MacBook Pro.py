# %%
from linearmodels.iv import compare
from linearmodels.iv import IV2SLS
import scipy.optimize as opt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from statsmodels.formula.api import ols
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import statsmodels.api as sm

# %%
# base_directory = 'K:\\OneDrive\\Insurance\\Data\\raw\\'
base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'

os.chdir(base_directory)

data = []
with open(base_directory+'data_filtered_combined_s8_to_s17.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])
cols = ['loss_amount', 'round_number', 'purchased_coverage',
        'random_number', 'timer_faq_choiceaffectsprice',
        'timer_faq_howitworks', 'timer_faq_noinsurance', 'timer_faq_coverage',
        'timer_char7', 'timer_char4',
        'timer_char3', 'timer_char2', 'timer_char1', 'insurance', 'WTP',
        'belief', 'prob', 'var4', 'var3', 'var2', 'var1',
        'number_entered', 'result_risk', 'age',
        'gender', 'smoker', 'health', 'exercisedays', 'exercisehours',
        'insurancehealth', 'lifeinsurance', 'riskdriving', 'riskfinance',
        'risksport', 'riskjob', 'riskhealth', 'risk', 'final_payoff',
        'HL_switchpoint', 'HLLF_switchpoint', 'diff_belief_prob',
        'timer_all_chars']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']] = df[[
    'timer_char1', 'timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']].fillna(0)

df = df.drop(columns='timer_all_chars')

df['timer_all_chars'] = (df['timer_char1'] + df['timer_char2'] +
                         df['timer_char3'] + df['timer_char4'])/1000

df = df.reset_index()


# find variance for each probability decile
df['decile'] = pd.qcut(df['prob'], q=10, labels=False)
# %
data_belief_prob_treatment = pd.DataFrame(df.groupby(['decile']).agg({'diff_belief_prob': ['mean', 'var'],
                                                                      'belief': ['mean', 'var'],
                                                                      'prob': ['mean', 'var']})).reset_index()

# %
# what would the probabilities be if individuals only accounted for the
# positive or negative or variable info

prob_adjusted_for_treatment = []
with open('/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/treatment_prob_sum.csv') as csv_file:
    # with open('K:\\OneDrive\\Insurance\\treatment_prob_sum.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        prob_adjusted_for_treatment.append(row)

prob_adjusted_for_treatment[0][0] = 'true_prob'
# %
prob_adj_treat_df = pd.DataFrame(
    prob_adjusted_for_treatment[1:], columns=prob_adjusted_for_treatment[0])
cols = prob_adjusted_for_treatment[0]
prob_adj_treat_df[cols] = prob_adj_treat_df[cols].apply(
    pd.to_numeric, errors='coerce')
# %

prob_adj_treat_df['prob'] = list(set(df['prob'].tolist()))
# %
df = pd.merge(df, prob_adj_treat_df, how='left', on='prob').reset_index()
# %
df.drop(columns=['index'])
# %
diff_belief_prob_adj_treatment = []

for p in range(len(df)):
    if df.iloc[p]['treatment'] == 'neginfo':
        diff_belief_prob_adj_treatment.append(
            df.iloc[p]['belief'] - 100*df.iloc[p]['neg_prob'])
    if df.iloc[p]['treatment'] == 'posinfo':
        diff_belief_prob_adj_treatment.append(
            df.iloc[p]['belief'] - 100*df.iloc[p]['pos_prob'])
    if df.iloc[p]['treatment'] == 'varinfo':
        diff_belief_prob_adj_treatment.append(
            df.iloc[p]['belief'] - 100*df.iloc[p]['var_prob'])
    if df.iloc[p]['treatment'] == 'fullinfo':
        diff_belief_prob_adj_treatment.append(
            df.iloc[p]['belief'] - 100*df.iloc[p]['prob'])
    if df.iloc[p]['treatment'] == 'baseline':
        diff_belief_prob_adj_treatment.append(
            df.iloc[p]['belief'] - 100*df.iloc[p]['prob'])

df = df.assign(diff_belief_prob_adj_treatment=diff_belief_prob_adj_treatment)

df_adj_belief_prob_treatment = pd.DataFrame(df.groupby(
    ['treatment']).agg({'diff_belief_prob_adj_treatment': 'mean'})).reset_index()

# %
plt.bar(df_adj_belief_prob_treatment['treatment'].tolist(
), df_adj_belief_prob_treatment['diff_belief_prob_adj_treatment'].tolist())
plt.xlabel('Treatment')
plt.ylabel('Diff between belief and prob, adjusted')
plt.axhline(y=0, linewidth=1, color='k')
# plt.savefig(
# 'filtered_plots_s8_to_s17/diff_belief_prob_treatment_bars_adjusted.png')
# plt.savefig(
# 'filtered_plots_s8_to_s10\\diff_belief_prob_treatment_bars_adjusted.png')
plt.show()

# os.chdir(base_directory+'ols_s8_to_s17/')

# model_names = ['model_0_accuracy',
#                'model_1_accuracy',
#                'model_2_accuracy',
#                'model_3_accuracy',
#                'model_4_accuracy',
#                'model_5_belief',
#                'model_6_wtp',
#                'model_7_wtp',
#                'moodel_8_accuracy',
#                'model_9_compare_betas']

# model_summaries = [model_0_accuracy.summary(),
#                    model_1_accuracy.summary(),
#                    model_2_accuracy.summary(),
#                    model_3_accuracy.summary(),
#                    model_4_accuracy.summary(),
#                    model_5_belief.summary(),
#                    model_6_wtp.summary(),
#                    model_7_wtp.summary(),
#                    model_8_accuracy.summary(),
#                    model_9_compare_betas.summary()]

# for i in range(len(model_summaries)):
#     with open(model_names[i] + '.html', 'w') as fh:
#         sum_table = model_summaries[i].tables[0].as_html(
#         ) + '\n' + model_summaries[i].tables[1].as_html() + '\n' + model_summaries[i].tables[2].as_html()
#         fh.write(sum_table)
# %

# find variance by treatment
var_by_treatment_df = pd.DataFrame(df.groupby(['treatment']).agg({'diff_belief_prob': ['mean', 'var', 'count'],
                                                                  'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                  'belief': ['mean', 'var'],
                                                                  'prob': ['mean', 'var']})).reset_index()


# %

# %
print(str(6.3227/0.1441) + ';' + str(1.3897/0.0469) +
      ';' + str(3.0825/0.1067) + ';' + str(1.4568/0.0618))
# %
demo_info = df[['treatment', 'age', 'gender', 'smoker', 'health', 'exercisedays',
                'exercisehours', 'insurancehealth', 'lifeinsurance', 'riskdriving',
                'riskfinance', 'risksport', 'riskjob', 'riskhealth', 'risk']]

# %
demo_info_sum = pd.DataFrame(demo_info.groupby(
    ['treatment']).agg({'age': ['mean', 'var', 'count'],
                        'gender': ['mean', 'var', 'count'],
                        'smoker': ['mean', 'var', 'count'],
                        'health': ['mean', 'var', 'count'],
                        'exercisedays': ['mean', 'var', 'count'],
                        'exercisehours': ['mean', 'var', 'count'],
                        'insurancehealth': ['mean', 'var', 'count'],
                        'lifeinsurance': ['mean', 'var', 'count'],
                        'riskdriving': ['mean', 'var', 'count'],
                        'riskfinance': ['mean', 'var', 'count'],
                        'risksport': ['mean', 'var', 'count'],
                        'riskjob': ['mean', 'var', 'count'],
                        'riskhealth': ['mean', 'var', 'count'],
                        'risk': ['mean', 'var', 'count']})).reset_index()
# %
group_baseline = demo_info.loc[demo_info['treatment'] == 'baseline']
group_fullinfo = demo_info.loc[demo_info['treatment'] == 'fullinfo']
group_neginfo = demo_info.loc[demo_info['treatment'] == 'neginfo']
group_posinfo = demo_info.loc[demo_info['treatment'] == 'posinfo']
group_varinfo = demo_info.loc[demo_info['treatment'] == 'varinfo']
# %
demo_varnames = ['age', 'gender', 'smoker', 'health', 'exercisedays',
                 'exercisehours', 'insurancehealth', 'lifeinsurance', 'riskdriving',
                 'riskfinance', 'risksport', 'riskjob', 'riskhealth', 'risk']

# %
groups = [group_baseline, group_fullinfo,
          group_neginfo, group_posinfo, group_varinfo]
groups_tteststat = [groups]
for varname in demo_varnames:
    var_group = []
    group_index = [0, 1, 2, 3, 4]
    for g in range(len(groups)):
        group = groups[g]
        group_tteststat = []
        # group_index = group_index.remove(g)
        for ng in group_index:
            other_group = groups[ng]
            pvalue = ttest_ind(group[varname], other_group[varname])[1]
            group_tteststat.append(pvalue)
            # index 1 for pvalue
        # group_index.append(g)
        var_group.append(group_tteststat)
    groups_tteststat.append(var_group)
# %
tteststats_all_vars = groups_tteststat[1:]
# # %%

# %%

# %

features_of_interest = ['loss_amount', 'round_number', 'purchased_coverage',
                        'random_number', 'state', 'timer_faq_choiceaffectsprice',
                        'timer_faq_howitworks', 'timer_faq_noinsurance', 'timer_faq_coverage',
                        'timer_char7', 'timer_char4', 'timer_char3', 'timer_char2',
                        'timer_char1', 'insurance', 'WTP', 'belief', 'prob', 'var4', 'var3',
                        'var2', 'var1', 'number_entered', 'result_risk',
                        'age', 'gender', 'smoker', 'health', 'exercisedays',
                        'exercisehours', 'insurancehealth', 'lifeinsurance', 'riskdriving',
                        'riskfinance', 'risksport', 'riskjob', 'riskhealth', 'risk',
                        'final_payoff', 'HL_switchpoint', 'HLLF_switchpoint',
                        'diff_belief_prob', 'timer_all_chars', 'decile', 'true_prob',
                        'var_prob', 'neg_prob', 'pos_prob', 'diff_belief_prob_adj_treatment', 'treatment']

df_quantifiable = df.loc[:, features_of_interest]

df_quantifiable[features_of_interest] = df_quantifiable[features_of_interest].apply(
    pd.to_numeric, errors='coerce')
df_quantifiable.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
# %
vif = pd.DataFrame()
vif['Features'] = df_quantifiable.columns
vif['VIF'] = [variance_inflation_factor(
    df_quantifiable.values, i) for i in range(df_quantifiable.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)
vif

# # %%
# model_posinfo = ols(
#     'abs(diff_belief_prob) ~ prob + loss_amount + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='posinfo']).fit(cov_type="HC0")

# model_neginfo = ols(
#     'abs(diff_belief_prob) ~ prob + loss_amount + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='neginfo']).fit(cov_type="HC0")

# model_varinfo = ols(
#     'abs(diff_belief_prob) ~ prob + loss_amount + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='varinfo']).fit(cov_type="HC0")

# model_fullinfo = ols(
#     'abs(diff_belief_prob) ~ prob + loss_amount + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='fullinfo']).fit(cov_type="HC0")

# model_baseline = ols(
#     'abs(diff_belief_prob) ~ prob + loss_amount + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='baseline']).fit(cov_type="HC0")

# os.chdir(base_directory+'ols_s8_to_s17/')

# model_names_var_treatment = ['model_posinfo',
#                    'model_neginfo',
#                    'model_varinfo',
#                    'model_fullinfo',
#                    'model_baseline']

# model_summaries_var_treatment = [model_posinfo.summary(),
#                        model_neginfo.summary(),
#                        model_varinfo.summary(),
#                        model_fullinfo.summary(),
#                        model_baseline.summary(),]

# for i in range(len(model_summaries_var_treatment)):
#     with open(model_names_var_treatment[i] + '.html', 'w') as fh:
#         sum_table = model_summaries_var_treatment[i].tables[0].as_html(
#         ) + '\n' + model_summaries_var_treatment[i].tables[1].as_html() + '\n' + model_summaries_var_treatment[i].tables[2].as_html()
#         fh.write(sum_table)
# %%
# summary statistics of characteristic 1
df['timer_char1'].describe()
# df = df.drop(columns='index')
# %
# get unique labels = 187 unique labels = 187 participants
df['label'].unique()

# create absolute difference in belief and prob
df['abs_diff_belief_prob'] = abs(df['diff_belief_prob'])
# %
# group by participants, and find mean difference between belief and prob
belief_prob_by_participant = pd.DataFrame(df.groupby(['label'], as_index=False).agg({'diff_belief_prob': ['mean', 'var'],
                                                                                     'belief': ['mean', 'var'],
                                                                                     'prob': ['mean', 'var'],
                                                                                     'abs_diff_belief_prob': ['mean', 'var']}))  # .reset_index()


# %
# convert to list
belief_prob_by_participant_list = belief_prob_by_participant.values.tolist()
# %

# create a ranking of participants based on how "accurate" they are
belief_prob_by_participant[['accuracy_rank_mean', 'accuracy_rank_var']
                           ] = belief_prob_by_participant['abs_diff_belief_prob'].rank(method='average', ascending=True)


# %
# group by participants and for each participant (within participant)
# rank the accuracy
# so one participant goes through 5 rounds
# see in which round (which profile/which prob) leads individuals to be most accurate

# regression: is there a relationship between probability and ranking?
# like for higher probabilities, do people tend to be more accurate or less accurate
df['accuracy_rank_round_by_participant'] = df.groupby(
    ['label'])['abs_diff_belief_prob'].rank()

# similarly, rank the over/understimation
# like for higher probabilities, do people tend to over or underestimate?
df['diff_rank_round_by_participant'] = df.groupby(
    'label')['diff_belief_prob'].rank()

# regression: is there a relationship between probability and ranking?

# %
# instead of grouping by participants, try grouping by (unique) probabilities

df['accuracy_rank_round_by_prob'] = df.groupby(
    ['prob'])['abs_diff_belief_prob'].rank()

# similarly, rank the over/understimation
# like for higher probabilities, do people tend to over or underestimate?
df['diff_rank_round_by_prob'] = df.groupby('prob')['diff_belief_prob'].rank()

# #%
# plt.plot(df['prob'], df['accuracy_rank_round_by_prob'], '.')
# plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
# plt.xlabel('Prob')
# plt.ylabel('Ranking')
# plt.title('Accuracy ranking')
# plt.show()

# plt.plot(df['prob'], df['diff_rank_round_by_prob'], '.')
# plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
# plt.xlabel('Prob')
# plt.ylabel('Ranking')
# plt.title('Diff belief and prob ranking')
# plt.show()
# %
# now create a ranking from biggest underestimator to biggest overestimator
# should the ranking method be average
# some probabilities are inevitably higher or lower
# do people tend to overestimate or underestimate?
# average is fine

belief_prob_by_participant[['estimation_rank_mean', 'estimation_rank_var']
                           ] = belief_prob_by_participant['diff_belief_prob'].rank(method='average', ascending=True)

# %
# belief_prob_by_participant = belief_prob_by_participant.drop(columns='index')

# %
# merge the participant rankings with the original dataframe
df = pd.merge(df, belief_prob_by_participant,
              how='left', on='label')
# df = df.drop(columns='index')
# %

# find variance of accuracy by treatment
# find variance by treatment
var_by_treatment_df = pd.DataFrame(df.groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                  'diff_belief_prob': ['mean', 'var', 'count'],
                                                                  'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                  'belief': ['mean', 'var'],
                                                                  'prob': ['mean', 'var']})).reset_index()
# %
# find how many people overestimated their risk in each treatment
moments_of_overestimators_df = pd.DataFrame(df.loc[df['diff_belief_prob'] > 0].groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                           'diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                          'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                                                           'belief': ['mean', 'var'],
                                                                                                           'prob': ['mean', 'var']})).reset_index()

# %
# find how many people underestimated their risk in each treatment
moments_of_underestimators_df = pd.DataFrame(df.loc[df['diff_belief_prob'] < 0].groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                            'diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                           'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                                                            'belief': ['mean', 'var'],
                                                                                                            'prob': ['mean', 'var']})).reset_index()

# %
moments_under_overestimators_df = pd.merge(moments_of_overestimators_df[['treatment', 'diff_belief_prob']],
                                           moments_of_underestimators_df[[
                                               'treatment', 'diff_belief_prob']],
                                           how='inner', on='treatment').reset_index()

moments_under_overestimators_df = moments_under_overestimators_df.drop(
    columns='index')
# %
print(moments_under_overestimators_df.to_latex(index=False))
os.chdir(base_directory)
# construct histogram of accuracy (abs diff btw belief and prob)
df['abs_diff_belief_prob'].hist(by=df['treatment'])
# plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_treatment_histogram.png')
# plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_treatment_histogram.png')
plt.show()

# %%

# sum stats PER unique probability
# variance by true probability
var_by_prob_df = pd.DataFrame(df.groupby(['prob']).agg({'abs_diff_belief_prob': ['mean', 'var'],
                                                        'diff_belief_prob': ['mean', 'var'],
                                                        'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                        'belief': ['mean', 'var'],
                                                        'prob': ['mean']})).reset_index()

var_by_prob_df[['prob', 'abs_diff_belief_prob']]
# %%

# plot data - mean
plt.plot(var_by_prob_df['prob']['mean'],
         var_by_prob_df['abs_diff_belief_prob']['mean'], '.', label='Data')
plt.xlabel('Prob')
plt.ylabel('Abs diff between belief and probability')

# This is the function we are trying to fit to the data.


def func(x, a, b, c):
    return a * x*x + b*x + c


# fit a curve to data
optimizedParameters, pcov = opt.curve_fit(
    func, var_by_prob_df['prob']['mean'], var_by_prob_df['abs_diff_belief_prob']['mean'])
plt.plot(var_by_prob_df['prob']['mean'], func(
    var_by_prob_df['prob']['mean'], *optimizedParameters), label='Fit')
# plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob.png')
# plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob.png')
plt.show()

# plot data - var
plt.plot(var_by_prob_df['prob']['mean'],
         var_by_prob_df['abs_diff_belief_prob']['var'], '.', label='Data')
plt.xlabel('Prob')
plt.ylabel('Abs diff between belief and probability (var)')
# plt.savefig(
#     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_var.png')
# plt.savefig(
#     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var.png')
plt.show()
# %%

# treatment_names = ['baseline', 'fullinfo', 'neginfo', 'posinfo', 'varinfo']
# var_by_prob_df_container = {}
# markers = ['.', 's', 'v', '^', '1']
# marker_size = [6, 3, 5, 5, 7]
# colors = ['darkgray', 'black', 'red', 'green', 'blue']

treatment_names = ['fullinfo', 'baseline']
var_by_prob_df_container = {}
markers = ['s', '.']
marker_size = [3, 6]
colors = ['black', 'darkgray']


marker_count = 0
plt.clf()
for treatment_name in treatment_names:
    var_by_prob_treatment_df = pd.DataFrame(df.loc[df['treatment'] == treatment_name].groupby(['prob']).agg(
        {'abs_diff_belief_prob': ['mean', 'var'],
         'diff_belief_prob': ['mean', 'var'],
         'diff_belief_prob_adj_treatment': ['mean', 'var'],
         'belief': ['mean', 'var'],
         'prob': ['mean']})).reset_index()
    var_by_prob_df_container.update({treatment_name: var_by_prob_treatment_df})
    # # plot data - mean
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['mean'], markers[marker_count], markersize=marker_size[marker_count], mfc=colors[marker_count],mec=colors[marker_count], label=treatment_name)
    # plt.xlabel('Prob')
    # # plt.ylabel('Abs diff between belief and probability, ' + treatment_name)
    # plt.ylabel('Abs diff between belief and probability by treatment ')
    # # fit curve to data
    # optPar, pcov_treat = opt.curve_fit(func, var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['mean'])
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], func(var_by_prob_treatment_df['prob']['mean'], *optPar), color=colors[marker_count])#,label='Fit')
    # plt.legend(loc=(0.45,0.70)) #(loc=(x,y))
    # # plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
    # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
    # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_'+treatment_name+'.png')
    # # plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_'+treatment_name+'.png')
    # # plt.clf()
    # # plt.show()
    # plot data - var
    plt.plot(var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['var'], markers[marker_count],
             markersize=marker_size[marker_count], mfc=colors[marker_count], mec=colors[marker_count], label=treatment_name)
    plt.xlabel('Prob')
    # , ' + treatment_name)
    plt.ylabel('Abs diff between belief and probability (var)')
    # fit curve to data
    optPar, pcov_treat = opt.curve_fit(
        func, var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['var'])
    plt.plot(var_by_prob_treatment_df['prob']['mean'], func(
        var_by_prob_treatment_df['prob']['mean'], *optPar), color=colors[marker_count])  # label='Fit')
    plt.legend(loc=(0.45, 0.70))
    # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # # plt.clf()
    marker_count += 1

plt.show()

# %%
# get rid of condition number.
# try to center some variables

# df['prob'] -= np.average(df['prob'])
# df['diff_belief_prob'] -= np.average(df['diff_belief_prob'])
# df['belief'] -= np.average(df['belief'])
# df['abs_diff_belief_prob'] -= np.average(df['abs_diff_belief_prob'])

# #%%
# # basic accuracy model
# model_0_accuracy = ols(
#     'abs_diff_belief_prob ~ C(treatment) + prob + timer_all_chars + round_number', data=df).fit(cov_type="HC0")
# # %
# model_0_accuracy.summary()

# #%%
# # basic accuracy model with prob and prob^2
# model_1_accuracy = ols(
#     'abs_diff_belief_prob ~ C(treatment) + prob + timer_all_chars + round_number +' + 'I(prob**2)', data=df).fit(cov_type="HC0")
# # %
# model_1_accuracy.summary()
# %%

# filter dataset
# exclude people who spent less than 10 ms or more than 200000 ms
# on each characteristic

filtered_df = df.loc[
    (df['timer_char1'] > 10) &
    (df['timer_char1'] < 200000) &
    (df['timer_char2'] > 10) &
    (df['timer_char2'] < 200000) &
    (df['timer_char3'] > 10) &
    (df['timer_char3'] < 200000) &
    (df['timer_char4'] > 10) &
    (df['timer_char4'] < 200000)]

# center variables to avoid large condition number error
filtered_df['prob'] -= np.average(filtered_df['prob'])
# filtered_df['diff_belief_prob'] -= np.average(filtered_df['diff_belief_prob'])
# filtered_df['belief'] -= np.average(filtered_df['belief'])
filtered_df['abs_diff_belief_prob'] -= np.average(
    filtered_df['abs_diff_belief_prob'])

# %%

# basic accuracy model
model_0_accuracy_filtered = ols(
    'abs_diff_belief_prob ~   C(treatment) + prob + timer_all_chars + round_number + loss_amount', data=filtered_df).fit(cov_type="HC0")
# %
model_0_accuracy_filtered.summary()

# %%
# basic accuracy model with prob and prob^2
model_1_accuracy_filtered = ols(
    'abs_diff_belief_prob ~ C(treatment) + prob + timer_all_chars + round_number + loss_amount+' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")
# %
model_1_accuracy_filtered.summary()
# %%

# basic accuracy model with vars
model_2_accuracy_filtered = ols(
    'abs_diff_belief_prob  ~ C(treatment) + prob + timer_all_chars + var1 +var2 +var3 + var4 + round_number+ loss_amount ', data=filtered_df).fit(cov_type="HC0")
# %
model_2_accuracy_filtered.summary()

# %%
# basic accuracy model with prob and prob^2
model_3_accuracy_filtered = ols(
    'abs_diff_belief_prob ~ C(treatment) + prob + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount +' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")
# %
model_3_accuracy_filtered.summary()
# %%

# check how beliefs are biased in the basic model
# ie see if the diff is pos or neg for each treatment
model_4_diff = ols(
    'diff_belief_prob ~ C(treatment) + prob + timer_all_chars + round_number+ loss_amount + ' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")

model_4_diff.summary()

# no effect
# %%
# same as model 4 but with var

model_5_diff = ols(
    'diff_belief_prob ~ C(treatment) + prob + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount + ' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")

model_5_diff.summary()

# %%

filtered_df = df.loc[
    (df['timer_char1'] > 10) &
    (df['timer_char1'] < 200000) &
    (df['timer_char2'] > 10) &
    (df['timer_char2'] < 200000) &
    (df['timer_char3'] > 10) &
    (df['timer_char3'] < 200000) &
    (df['timer_char4'] > 10) &
    (df['timer_char4'] < 200000)]

# %%
# create ln belief/(1-belief)

filtered_df['belief_over_100'] = filtered_df['belief']/100
filtered_df['ln_belief'] = np.log(
    filtered_df['belief_over_100']/(1-filtered_df['belief_over_100']))
filtered_df['ln_prob'] = np.log(filtered_df['prob']/(1-filtered_df['prob']))

# %%
# get rid of observations with value of ln_belief == inf/-inf
# ie get rid of observations where belief_over_100 = 0 or 1

filtered_df = filtered_df.loc[
    (filtered_df['belief_over_100'] != 0) &
    (filtered_df['belief_over_100'] != 1)]


# %%
model_6_prob = ols(
    'ln_prob ~ C(treatment) + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount ', data=filtered_df).fit(cov_type="HC0")

model_6_prob.summary()

# %%
# same as model 6 but regress on dummies for each participant
model_7_prob = ols(
    'ln_prob ~ C(label) + C(treatment) + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount ', data=filtered_df).fit(cov_type="HC0")

model_7_prob.summary()

# %%
model_8_belief = ols(
    'ln_belief ~ C(treatment) + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount +' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")

model_8_belief.summary()
# %%
# regress on dummies for each participant
model_9_belief = ols(
    'ln_belief ~ C(label) + C(treatment) + timer_all_chars + var1 + var2 + var3 + var4 + round_number+ loss_amount + ' + 'I(prob**2)', data=filtered_df).fit(cov_type="HC0")

model_9_belief.summary()

# %%

os.chdir(base_directory+'ols_s8_to_s17/')

models_filtered_names = [
    'model_0_accuracy_filtered',
    'model_1_accuracy_filtered',
    'model_2_accuracy_filtered',
    'model_3_accuracy_filtered',
    'model_4_diff',
    'model_5_diff',
    'model_6_prob',
    'model_7_prob',
    'model_8_belief',
    'model_9_belief'
]

models_filtered = [
    model_0_accuracy_filtered,
    model_1_accuracy_filtered,
    model_2_accuracy_filtered,
    model_3_accuracy_filtered,
    model_4_diff,
    model_5_diff,
    model_6_prob,
    model_7_prob,
    model_8_belief,
    model_9_belief
]

for m in range(len(models_filtered)):
    with open(models_filtered_names[m] + '.txt', 'w') as f:
        for table in models_filtered[m].summary().tables:
            f.write(
                table.as_latex_tabular()
            )
    f.close()

# %%

# %%
# predict values for beliefs

model_10_belief = ols(
    'belief_over_100 ~ C(treatment) + var1 + var2 + var3 + var4 + round_number', data=filtered_df).fit(cov_type="HC0")

model_10_belief.summary()

# %%
# is willingness to pay logically affected by characteristics

# is willingness to pay logically affected by treatment
# -- treatment is the instrument

# is willingness to pay logically affected by loss amount - yes

model_wtp_2sls_first = IV2SLS(filtered_df['belief_over_100'], filtered_df[[
                              'treatment', 'var1', 'var2', 'var3', 'var4', 'loss_amount']], None, None).fit(cov_type="unadjusted")
print(model_wtp_2sls_first)

filtered_df['belief_hat'] = filtered_df['belief_over_100'] - \
    model_wtp_2sls_first.resids
# %%
model_wtp_2sls_second = IV2SLS(filtered_df['WTP'], filtered_df[['var1', 'var2', 'var3', 'var4', 'loss_amount']],
                               filtered_df['belief_over_100'], filtered_df['treatment']).fit(cov_type="unadjusted")

print(model_wtp_2sls_second)

# %%

model_wtp_ols = IV2SLS(filtered_df['WTP'], filtered_df[[
                       'belief_over_100', 'var1', 'var2', 'var3', 'var4', 'loss_amount']], None, None).fit(cov_type="unadjusted")

model_wtp_2sls_direct = IV2SLS(filtered_df['WTP'], filtered_df[[
                               'belief_hat', 'var1', 'var2', 'var3', 'var4', 'loss_amount']], None, None).fit(cov_type="unadjusted")

print(compare({'OLS': model_wtp_ols,
      '2SLS': model_wtp_2sls_second, 'Direct': model_wtp_2sls_direct}))

# %%
