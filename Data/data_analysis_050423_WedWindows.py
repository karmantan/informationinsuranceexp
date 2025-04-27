# %%
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
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
base_directory = 'U:\\documents\\Chapter_2\\Data\\raw\\'
# base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'

os.chdir(base_directory)

data = []
# with open(base_directory+'data_filtered_combined_s8_to_s17.csv') as csv_file:
with open(base_directory+'data_combined_s8_to_s17.csv') as csv_file:
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

timer_cols = ['timer_char1', 'timer_char2',
              'timer_char3', 'timer_char4', 'timer_char7']

# %%
# replace glitch timers with average time spent
participants_with_glitch_timer = df[[
    'label', 'round_number']].loc[df['timer_char1'] > 200000].values.tolist()
for participant, round in participants_with_glitch_timer:
    average_time = sum(df[['timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']].loc[(df['label'] == participant) & (df['round_number'] == round)].values.tolist()[
                       0])/len(df[['timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']].loc[(df['label'] == participant) & (df['round_number'] == round)].values.tolist()[0])
    df['timer_char1'].mask((df['label'] == participant) & (
        df['round_number'] == round), average_time, inplace=True)
    # df[timer_col].mask(df[timer_col] > 200000, 0, inplace=True)
# %%
# participant with timer 7 glitch
average_time_participant_timer_7_glitch = sum(df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4']].loc[df['timer_char7'] > 200000].values.tolist()[
                                              0])/len(df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4']].loc[df['timer_char7'] > 200000].values.tolist()[0])
df['timer_char7'].mask(df['timer_char7'] > 200000,
                       average_time_participant_timer_7_glitch, inplace=True)


# %%
for timer_col in timer_cols:
    df[timer_col] = df[timer_col]/(1000)

df['timer_all_chars'] = (df['timer_char1'] + df['timer_char2'] +
                         df['timer_char3'] + df['timer_char4'] + df['timer_char7'])

# for timer_col in timer_cols:
#     # recenter timers
#     df[timer_col] -= np.average(df[timer_col])

# # recenter timer_all_chars
# df['timer_all_chars'] -= np.average(df['timer_all_chars'])
# df = df.reset_index()

# %%
df['abs_diff_belief_prob'] = abs(df['diff_belief_prob'])
# %
# find variance by treatment
var_by_treatment_df = pd.DataFrame(df.groupby(['treatment']).agg({'diff_belief_prob': ['mean', 'var', 'count'],
                                                                  #   'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                  'belief': ['mean', 'var'],
                                                                  'prob': ['mean', 'var']})).reset_index()


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
                        'var_prob', 'neg_prob', 'pos_prob']  # , 'diff_belief_prob_adj_treatment', 'treatment']


# %


# find variance of accuracy by treatment
# find variance by treatment
var_by_treatment_df = pd.DataFrame(df.groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                  'diff_belief_prob': ['mean', 'var', 'count'],
                                                                  # 'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                  'belief': ['mean', 'var'],
                                                                  'prob': ['mean', 'var']})).reset_index()
# %
# find how many people overestimated their risk in each treatment
moments_of_overestimators_df = pd.DataFrame(df.loc[df['diff_belief_prob'] > 0].groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                           'diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                          # 'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                                                           'belief': ['mean', 'var'],
                                                                                                           'prob': ['mean', 'var']})).reset_index()

# %
# find how many people underestimated their risk in each treatment
moments_of_underestimators_df = pd.DataFrame(df.loc[df['diff_belief_prob'] < 0].groupby(['treatment']).agg({'abs_diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                            'diff_belief_prob': ['mean', 'var', 'count'],
                                                                                                           # 'diff_belief_prob_adj_treatment': ['mean', 'var'],
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
plt.savefig('plots_s8_to_s17/accuracy_abs_diff_treatment_histogram.png')
plt.show()

# %

# sum stats PER unique probability
# variance by true probability
var_by_prob_df = pd.DataFrame(df.groupby(['prob']).agg({'abs_diff_belief_prob': ['mean', 'var'],
                                                        'diff_belief_prob': ['mean', 'var'],
                                                        # 'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                        'belief': ['mean', 'var'],
                                                        'prob': ['mean']})).reset_index()

var_by_prob_df[['prob', 'abs_diff_belief_prob']]
# %

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
# %

# %%
treatment_names = ['baseline', 'fullinfo', 'neginfo', 'posinfo', 'varinfo']
var_by_prob_df_container = {}
markers = ['.', 's', 'v', '^', '1']
marker_size = [6, 3, 5, 5, 7]
colors = ['darkgray', 'black', 'red', 'green', 'blue']

# treatment_names = ['fullinfo', 'baseline']
# var_by_prob_df_container = {}
# markers = ['s', '.']
# marker_size = [3, 6]
# colors = ['black', 'darkgray']


marker_count = 0
plt.clf()
for treatment_name in treatment_names:
    var_by_prob_treatment_df = pd.DataFrame(df.loc[df['treatment'] == treatment_name].groupby(['prob']).agg(
        {'abs_diff_belief_prob': ['mean', 'var'],
         'diff_belief_prob': ['mean', 'var'],
         # 'diff_belief_prob_adj_treatment': ['mean', 'var'],
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
    # # plt.savefig('plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
    # plt.savefig('plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
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
    # plt.savefig('plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # plt.savefig(
    #     'plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # # plt.clf()
    marker_count += 1

plt.show()
#

# %%
# plot histogram of timers in all treatments
for timer in timer_cols:
    df[timer].hist(by=df['treatment'], label=timer)

# %%
bins_1 = np.linspace(0, 100, 50)
for treatment in treatment_names:
    plt.hist(df['timer_char1'].loc[df['treatment'] == treatment],
             bins_1, alpha=0.5, label=treatment)
plt.title('Characteristic 1 timer')
plt.legend(loc='upper right')
plt.show()

# %%
timer_sum_stats = pd.DataFrame(df.groupby(['treatment']).agg({'timer_char1': ['mean', 'var', 'count'],
                                                              'timer_char2': ['mean', 'var', 'count'],
                                                              'timer_char3': ['mean', 'var', 'count'],
                                                              'timer_char4': ['mean', 'var', 'count'],
                                                              'timer_char7': ['mean', 'var', 'count'],
                                                              'timer_all_chars': ['mean', 'var', 'count']})).reset_index()

# #%%
# relevant_mean_columns = []
# for column in timer_sum_stats.columns:
#     if column[1] == 'mean':
#         relevant_mean_columns.append(column)

# sum_timers = {}
# for i in range(len(treatment_names)):
#     timers_treatment = []
#     for mean in relevant_mean_columns:
#         timers_treatment.append(timer_sum_stats[mean][i])
#     sum_timers[i] = sum(timers_treatment)


# %%

# ##############################################################
# # center variables to avoid large condition number error
# df['prob'] -= np.average(df['prob'])
# df['diff_belief_prob'] -= np.average(df['diff_belief_prob'])
# df['belief'] -= np.average(df['belief'])
# df['abs_diff_belief_prob'] -= np.average(df['abs_diff_belief_prob'])

# %
# basic accuracy model
model_0_accuracy = ols(
    'abs_diff_belief_prob ~   C(treatment) +  timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 +  round_number + loss_amount', data=df).fit(cov_type="HC0")
# %
model_0_accuracy.summary()

# %%
model_0_accuracy_baseline = ols(
    'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number + loss_amount', data=df.loc[df['treatment'] == 'baseline']).fit(cov_type="HC0")

model_0_accuracy_baseline.summary()

# %%

model_0_accuracy_fullinfo = ols(
    'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number + loss_amount', data=df.loc[df['treatment'] == 'fullinfo']).fit(cov_type="HC0")

model_0_accuracy_fullinfo.summary()

# %%
model_0_accuracy_neginfo = ols(
    'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number + loss_amount', data=df.loc[df['treatment'] == 'neginfo']).fit(cov_type="HC0")

model_0_accuracy_neginfo.summary()

# %%
model_0_accuracy_posinfo = ols(
    'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number + loss_amount', data=df.loc[df['treatment'] == 'posinfo']).fit(cov_type="HC0")

model_0_accuracy_posinfo.summary()

# %%
model_0_accuracy_varinfo = ols(
    'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number + loss_amount', data=df.loc[df['treatment'] == 'posinfo']).fit(cov_type="HC0")

model_0_accuracy_varinfo.summary()


# %%

model_0_names = ['model_0_accuracy_baseline', 'model_0_accuracy_fullinfo', 'model_0_accuracy_neginfo',
                 'model_0_accuracy_posinfo', 'model_0_accuracy_varinfo', 'model_0_accuracy']
model_0_models = [model_0_accuracy_baseline, model_0_accuracy_fullinfo,
                  model_0_accuracy_neginfo, model_0_accuracy_posinfo, model_0_accuracy_varinfo, model_0_accuracy]


print(summary_col(model_0_models, stars=True, float_format='%0.2f'))

os.chdir(base_directory + '\\ols_s8_to_s17\\')
with open('models_0_accuracy.txt', 'w') as f:
    f.write(summary_col(model_0_models, stars=True,
            float_format='%0.2f').as_latex())
    f.close()

# %%
# basic accuracy model with vars

model_2_accuracy = ols(
    'abs_diff_belief_prob ~   C(treatment) + var1 + var2 + var3 + var4 +  timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 +  round_number + loss_amount', data=df).fit(cov_type="HC0")
# %
model_2_accuracy.summary()

# %%
model_2_accuracy_baseline = ols(
    'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number+ loss_amount ', data=df.loc[df['treatment'] == 'baseline']).fit(cov_type="HC0")
# %
model_2_accuracy_baseline.summary()


# %%
model_2_accuracy_fullinfo = ols(
    'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number+ loss_amount ', data=df.loc[df['treatment'] == 'fullinfo']).fit(cov_type="HC0")
# %
model_2_accuracy_fullinfo.summary()

# %%
model_2_accuracy_neginfo = ols(
    'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number+ loss_amount ', data=df.loc[df['treatment'] == 'neginfo']).fit(cov_type="HC0")
# %
model_2_accuracy_neginfo.summary()

# %%
model_2_accuracy_posinfo = ols(
    'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number+ loss_amount ', data=df.loc[df['treatment'] == 'posinfo']).fit(cov_type="HC0")
# %
model_2_accuracy_posinfo.summary()

# %%
model_2_accuracy_varinfo = ols(
    'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + round_number+ loss_amount ', data=df.loc[df['treatment'] == 'varinfo']).fit(cov_type="HC0")
# %
model_2_accuracy_varinfo.summary()

# %%
model_2_names = ['model_2_accuracy_baseline', 'model_2_accuracy_fullinfo', 'model_2_accuracy_neginfo',
                 'model_2_accuracy_posinfo', 'model_2_accuracy_varinfo', 'model_2_accuracy']
model_2_models = [model_2_accuracy_baseline, model_2_accuracy_fullinfo,
                  model_2_accuracy_neginfo, model_2_accuracy_posinfo, model_2_accuracy_varinfo, model_2_accuracy]


print(summary_col(model_2_models, stars=True, float_format='%0.2f'))

os.chdir(base_directory + '\\ols_s8_to_s17\\')
with open('models_2_accuracy.txt', 'w') as f:
    f.write(summary_col(model_2_models, stars=True,
            float_format='%0.2f').as_latex())
    f.close()

# %%

# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint
# basic accuracy model + controls
model_0_accuracy_controls = ols(
    'abs_diff_belief_prob ~   C(treatment) + timer_all_chars + round_number + loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")

model_0_controls_models = []

model_0_controls_dictionary = {}

for treat in treatment_names:
    model_0_accuracy_controls_treat = ols(
        'abs_diff_belief_prob ~   timer_all_chars + round_number + loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    model_0_controls_models.append(model_0_accuracy_controls_treat)
    model_0_controls_dictionary[treat] = model_0_accuracy_controls_treat

model_0_controls_models.append(model_0_accuracy_controls)

os.chdir(base_directory + '\\ols_s8_to_s17\\')
with open('models_0_accuracy_controls.txt', 'w') as f:
    f.write(summary_col(model_0_controls_models,
            stars=True, float_format='%0.2f').as_latex())
    f.close()
# %%

# basic accuracy model with vars and controls
model_2_accuracy_controls = ols(
    'abs_diff_belief_prob  ~ C(treatment) + var1 +var2 +var3 + var4 + timer_all_chars + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk ', data=df).fit(cov_type="HC0")
# %
model_2_accuracy_controls.summary()


# %%
# create ln belief/(1-belief)

df['belief_over_100'] = df['belief']/100
df['ln_belief'] = np.log(df['belief_over_100']/(1-df['belief_over_100']))
df['ln_prob'] = np.log(df['prob']/(1-df['prob']))

# replace values of ln_belief == inf/-inf with none

# %
df['ln_belief'].mask(df['ln_belief'] == float('inf'), None, inplace=True)
df['ln_belief'].mask(df['ln_belief'] == -float('inf'), None, inplace=True)
# df['fake_var2'] = abs(
#     df['var2']/10 + np.random.normal(0, 0.1, len(df['var2'])))
# df['fake_var4'] = abs(
#     df['var4']/10 + np.random.normal(0, 0.1, len(df['var4'])))

# recall that for var 2 and 4
# information was presented in the following way
# for values between 0 and 2, risk increases by blabla

# create new var2 and var4 which reflects that
df['var2_cat'] = df['var2']
df['var4_cat'] = df['var4']

df['var2_cat'].mask(df['var2'] <= 2, 1, inplace=True)
df['var4_cat'].mask(df['var4'] <= 2, 1, inplace=True)

for r in range(1, 6):
    r_plus_one = r+1
    two_r = r*2
    two_r_plus_two = r_plus_one * 2
    df['var2_cat'].mask((df['var2'] > two_r) & (
        df['var2'] <= two_r_plus_two), r_plus_one, inplace=True)
    df['var4_cat'].mask((df['var4'] > two_r) & (
        df['var4'] <= two_r_plus_two), r_plus_one, inplace=True)
# %%

# %
model_5_prob = ols(
    'ln_prob ~ var1 + var2 + var3 + var4 - 1 ', data=df).fit(cov_type="HC0")

model_5_prob.summary()


# %%
model_6_belief_basic = ols(
    'ln_belief ~ var1 + var2 + var3 + var4-1', data=df).fit(cov_type="HC0")

model_6_belief_basic.summary()

# %%

model_7_belief = ols(
    'ln_belief ~ treatment + var1 + var2 + var3 + var4 + timer_all_chars + round_number+ loss_amount ', data=df).fit(cov_type="HC0")

model_7_belief.summary()

# %%
# model 7 plus participant dummies

model_8_belief = ols(
    'ln_belief ~ treatment +C(label) + var1 + var2 + var3 + var4 + timer_all_chars + round_number+ loss_amount ', data=df).fit(cov_type="HC0")

model_8_belief.summary()

# %%

# model 7 plus controls
# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint +
model_9_belief_controls = ols(
    'ln_belief ~ treatment + var1 + var2 + var3 + var4 + timer_all_chars + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk - 1', data=df).fit(cov_type="HC0")

model_9_belief_controls.summary()

# %%
# model 8 plus controls
model_10_belief_controls = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2 + var3 + var4 + timer_all_chars + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk - 1', data=df).fit(cov_type="HC0")

model_10_belief_controls.summary()

# %%
df[['coef_var1', 'coef_var2', 'coef_var3', 'coef_var4',
    'beta_1', 'beta_2', 'beta_3', 'beta_4']] = [0, 0, 0, 0, 0, 0, 0, 0]

# %
# construct difference between participant coefficients and true coefficients
df[['diff_var1', 'diff_var2',
    'diff_var3', 'diff_var4']] = [0, 0, 0, 0]

# %
# regress on each participant

# # keep only participants which appear at least twice
# p_counts = df['label'].value_counts()
# df = df[df.label.isin(p_counts.index[p_counts.gt(1)])]
participant_labels = df['label'].unique().tolist()

coef_var_names = ['coef_var1', 'coef_var2', 'coef_var3', 'coef_var4']
beta_var_names = ['beta_1', 'beta_2', 'beta_3', 'beta_4']
diff_var_names = ['diff_var1', 'diff_var2', 'diff_var3', 'diff_var4']

for participant in participant_labels:
    participant_data = df.loc[df['label'] == participant]
    model_belief_participant = ols(
        'ln_belief ~ var1 + var2 + var3 + var4 - 1', data=participant_data).fit(cov_type='HC0')
    model_prob_participant = ols(
        'ln_prob ~ var1 + var2 + var3 + var4 - 1', data=participant_data).fit(cov_type='HC0')
    # model_belief_participant.summary()
    participant_betas = model_belief_participant.params.tolist()
    true_betas = model_prob_participant.params.tolist()
    for i in range(len(participant_betas)):
        df[coef_var_names[i]].loc[df['label'] ==
                                  participant] = participant_betas[i]
        df[beta_var_names[i]].loc[df['label'] == participant] = true_betas[i]
        df[diff_var_names[i]].loc[df['label'] ==
                                  participant] = participant_betas[i]-true_betas[i]

# %%
df = df.drop(columns=['index'])
# %%
# Mann Whitney U test between coefficients
participant_betas_name = ['coef_var1', 'coef_var2', 'coef_var3', 'coef_var4']
true_betas_name = ['beta_1', 'beta_2', 'beta_3', 'beta_4']
treatment_names = ['baseline', 'fullinfo', 'neginfo', 'posinfo', 'varinfo']

mannwhitneyu_results = {}
wilcoxon_results = {}
for treat in treatment_names:
    df_one_treatment = df.loc[df['treatment'] == treat]
    stat_p = []
    wilcoxon_stat_p = []
    for i in range(len(participant_betas_name)):
        part_beta_name = participant_betas_name[i]
        true_beta_name = true_betas_name[i]
        stat_i, p_i = mannwhitneyu(
            df_one_treatment[part_beta_name], df_one_treatment[true_beta_name])
        stat_p.append([stat_i, p_i])
        wilcoxon_res = wilcoxon(
            df_one_treatment[part_beta_name]-df_one_treatment[true_beta_name])
        wilcoxon_stat_p.append([wilcoxon_res.statistic, wilcoxon_res.pvalue])
    mannwhitneyu_results[treat] = stat_p
    wilcoxon_results[treat] = wilcoxon_stat_p


# print('Statistics=%.3f, p=%.3f' % (stat_1, p_1))
# print('Statistics=%.3f, p=%.3f' % (stat_2, p_2))
# print('Statistics=%.3f, p=%.3f' % (stat_3, p_3))
# print('Statistics=%.3f, p=%.3f' % (stat_4, p_4))

# %%
models_coefficients = {}
for i in range(4):
    j = i+1
    # how do different treatments affect the coefficient for variable 1,2,3,4?
    model_var_x = ols('coef_var' + str(j) +
                      ' ~ C(treatment) + round_number+ loss_amount ', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_' + str(j)] = model_var_x
    # how do different treatments affect the difference between the participant beta and true beta for each variable?
    model_var_diff_x = ols('diff_var' + str(j) +
                           ' ~ C(treatment) ', data=df).fit(cov_type="HC0")
    # no intercept
    model_var_diff_x_no_const = ols(
        'diff_var' + str(j) + ' ~ C(treatment) - 1', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_diff_' +
                        str(j) + '_no_const'] = model_var_diff_x_no_const
    # how treatments affect diff between participant beta and true beta with controls
    model_var_diff_x_controls = ols(
        'diff_var' + str(j) + ' ~ C(treatment) + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_diff_' +
                        str(j) + '_controls'] = model_var_diff_x_controls
    # no intercept
    model_var_diff_x_controls_no_const = ols(
        'diff_var' + str(j) + ' ~ C(treatment) + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk -1', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_diff_' +
                        str(j) + '_controls_no_const'] = model_var_diff_x_controls_no_const
    # how different treatments affect difference with participant fixed effects
    model_var_diff_x_part = ols(
        'diff_var' + str(j) + ' ~ C(treatment) + C(label)', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_diff_' +
                        str(j) + '_part'] = model_var_diff_x_part
    # how treatments affect difference with fixed effects and controls
    model_var_diff_x_controls_part = ols(
        'diff_var' + str(j) + ' ~ C(treatment) + C(label) + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")
    models_coefficients['model_var_diff_' +
                        str(j) + '_controls_part'] = model_var_diff_x_controls_part
# # %%
# os.chdir(base_directory+'ols_s8_to_s17/')

# models_coefs_names = [
#     'model_var_1', 'model_var_2', 'model_var_3', 'model_var_4',
#     'model_var_diff_1', 'model_var_diff_2', 'model_var_diff_3', 'model_var_diff_4'
# ]

# models_coefs = [
#     model_var_1, model_var_2, model_var_3, model_var_4,
#     model_var_diff_1, model_var_diff_2, model_var_diff_3, model_var_diff_4
# ]

# for m in range(len(models_coefs)):
#     with open(models_coefs_names[m] + '.txt', 'w') as f:
#         for table in models_coefs[m].summary().tables:
#             f.write(
#                 table.as_latex_tabular()
#             )
#     f.close()

# # %%

# %%
