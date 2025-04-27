# %%
from statsmodels.tools.tools import add_constant
from linearmodels.iv.results import compare
import scipy.stats as stats
from statsmodels.formula.api import probit as probit
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
from statsmodels.iolib.summary2 import summary_col

# %
# base_directory = 'K:\\OneDrive\\Insurance\\Data\\raw\\'
base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'
# base_directory = 'U:\\documents\\Chapter_2\\Data\\raw\\'
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

# %
# replace glitch timers with average time spent
participants_with_glitch_timer = df[[
    'label', 'round_number']].loc[df['timer_char1'] > 200000].values.tolist()
for participant, round in participants_with_glitch_timer:
    average_time = sum(df[['timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']].loc[(df['label'] == participant) & (df['round_number'] == round)].values.tolist()[
                       0])/len(df[['timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']].loc[(df['label'] == participant) & (df['round_number'] == round)].values.tolist()[0])
    df['timer_char1'].mask((df['label'] == participant) & (
        df['round_number'] == round), average_time, inplace=True)
    # df[timer_col].mask(df[timer_col] > 200000, 0, inplace=True)
# %
# participant with timer 7 glitch
average_time_participant_timer_7_glitch = sum(df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4']].loc[df['timer_char7'] > 200000].values.tolist()[
                                              0])/len(df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4']].loc[df['timer_char7'] > 200000].values.tolist()[0])
df['timer_char7'].mask(df['timer_char7'] > 200000,
                       average_time_participant_timer_7_glitch, inplace=True)


# %
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

# %
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
# # %
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
# plt.savefig('plots_s8_to_s17/accuracy_abs_diff_treatment_histogram.png')
# plt.show()

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
# plt.show()

# plot data - var
plt.plot(var_by_prob_df['prob']['mean'],
         var_by_prob_df['abs_diff_belief_prob']['var'], '.', label='Data')
plt.xlabel('Prob')
plt.ylabel('Abs diff between belief and probability (var)')
# plt.savefig(
#     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_var.png')
# plt.savefig(
#     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var.png')
# plt.show()
# %

# %
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

    # # plot data - diff between belief and prob not abs
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['diff_belief_prob']['mean'], markers[marker_count],
    #          markersize=marker_size[marker_count], mfc=colors[marker_count], mec=colors[marker_count], label=treatment_name)
    # plt.xlabel('Prob')
    # # plt.ylabel('Abs diff between belief and probability, ' + treatment_name)
    # plt.ylabel('Diff between belief and probability by treatment ')
    # # fit curve to data
    # optPar, pcov_treat = opt.curve_fit(
    #     func, var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['diff_belief_prob']['mean'])
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], func(
    #     var_by_prob_treatment_df['prob']['mean'], *optPar), color=colors[marker_count])  # ,label='Fit')
    # plt.legend(loc=(0.45, 0.70))  # (loc=(x,y))
    # plt.axhline(y=0,xmin=-0.3,xmax=1.3, linestyle='--', lw=0.7, mfc='black', mec='black')
    # # plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
    # # plt.savefig('plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment.png')
    # # plt.savefig('plots_s8_to_s17/diff_belief_prob_by_prob_and_treatment.png')
    # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_'+treatment_name+'.png')
    # # plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_'+treatment_name+'.png')
    # # plt.clf()
    # # plt.show()

    # # plot data - var
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['var'], markers[marker_count],
    #          markersize=marker_size[marker_count], mfc=colors[marker_count], mec=colors[marker_count], label=treatment_name)
    # plt.xlabel('Prob')
    # # , ' + treatment_name)
    # plt.ylabel('Abs diff between belief and probability (var)')
    # # fit curve to data
    # optPar, pcov_treat = opt.curve_fit(
    #     func, var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['abs_diff_belief_prob']['var'])
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], func(
    #     var_by_prob_treatment_df['prob']['mean'], *optPar), color=colors[marker_count])  # label='Fit')
    # plt.legend(loc=(0.45, 0.70))
    # # plt.savefig('plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # # plt.savefig(
    # # #     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # # plt.savefig(
    # # #     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # plt.savefig(
    # #     'plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # # # plt.clf()

    # plot data - var diff belief prob not abs
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['diff_belief_prob']['var'], markers[marker_count],
    #          markersize=marker_size[marker_count], mfc=colors[marker_count], mec=colors[marker_count], label=treatment_name)
    # plt.xlabel('Prob')
    # # , ' + treatment_name)
    # plt.ylabel('Diff between belief and probability (var)')
    # # fit curve to data
    # optPar, pcov_treat = opt.curve_fit(
    #     func, var_by_prob_treatment_df['prob']['mean'], var_by_prob_treatment_df['diff_belief_prob']['var'])
    # plt.plot(var_by_prob_treatment_df['prob']['mean'], func(
    #     var_by_prob_treatment_df['prob']['mean'], *optPar), color=colors[marker_count])  # label='Fit')
    # plt.legend(loc=(0.45, 0.70))
    # plt.savefig('plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # plt.savefig('filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_var_'+treatment_name+'.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17/accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # # plt.savefig(
    # #     'filtered_plots_s8_to_s17\\accuracy_abs_diff_belief_prob_by_prob_and_treatment_var.png')
    # plt.savefig(
    #     'plots_s8_to_s17/diff_belief_prob_by_prob_and_treatment_var.png')
    # # # plt.clf()

    marker_count += 1

plt.show()
#

# %
# plot histogram of timers in all treatments
for timer in timer_cols:
    df[timer].hist(by=df['treatment'], label=timer)


bins_1 = np.linspace(0, 100, 50)
for treatment in treatment_names:
    plt.hist(df['timer_char1'].loc[df['treatment'] == treatment],
             bins_1, alpha=0.5, label=treatment)
plt.title('Characteristic 1 timer')
plt.legend(loc='upper right')
plt.show()

# %
timer_sum_stats = pd.DataFrame(df.groupby(['treatment']).agg({'timer_char1': ['mean', 'var', 'count'],
                                                              'timer_char2': ['mean', 'var', 'count'],
                                                              'timer_char3': ['mean', 'var', 'count'],
                                                              'timer_char4': ['mean', 'var', 'count'],
                                                              'timer_char7': ['mean', 'var', 'count'],
                                                              'timer_all_chars': ['mean', 'var', 'count']})).reset_index()

# %
# %
# basic accuracy model
model_0_accuracy = ols(
    'abs_diff_belief_prob ~   C(treatment) +  timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 +  round_number + loss_amount', data=df).fit(cov_type="HC0")
# %
# basic accuracy model with vars

model_2_accuracy = ols(
    'abs_diff_belief_prob ~   C(treatment) + var1 + var2 + var3 + var4 +  timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 +  round_number + loss_amount', data=df).fit(cov_type="HC0")

# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint
# basic accuracy model + controls
model_0_accuracy_controls = ols(
    'abs_diff_belief_prob ~   C(treatment) + timer_all_chars + round_number + loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")

# basic accuracy model with vars and controls
model_2_accuracy_controls = ols(
    'abs_diff_belief_prob  ~ C(treatment) + var1 +var2 +var3 + var4 + timer_all_chars + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk ', data=df).fit(cov_type="HC0")

model_0_accuracy_models = []
model_0_accuracy_models_dict = {}
model_2_accuracy_models = []
model_2_accuracy_models_dict = {}
model_0_accuracy_controls_models = []
model_0_accuracy_controls_models_dict = {}
model_2_accuracy_controls_models = []
model_2_accuracy_controls_models_dict = {}

for treat in treatment_names:
    model_0_accuracy_treat = ols(
        'abs_diff_belief_prob ~   timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number + loss_amount', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    model_0_accuracy_models.append(model_0_accuracy_treat)
    model_0_accuracy_models_dict[treat] = model_0_accuracy_treat
    model_2_accuracy_treat = ols('abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number+ loss_amount ',
                                 data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    model_2_accuracy_models.append(model_2_accuracy_treat)
    model_2_accuracy_models_dict[treat] = model_2_accuracy_treat
    model_0_accuracy_control_treat = ols(
        'abs_diff_belief_prob ~ timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number + loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    model_0_accuracy_controls_models.append(model_0_accuracy_control_treat)
    model_0_accuracy_controls_models_dict[treat] = model_0_accuracy_control_treat
    model_2_accuracy_control_treat = ols(
        'abs_diff_belief_prob  ~ var1 +var2 +var3 + var4 + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk ', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    model_2_accuracy_controls_models.append(model_2_accuracy_control_treat)
    model_2_accuracy_controls_models_dict[treat] = model_2_accuracy_control_treat

model_0_accuracy_models.append(model_0_accuracy)
model_2_accuracy_models.append(model_2_accuracy)
model_0_accuracy_controls_models.append(model_0_accuracy_controls)
model_2_accuracy_controls_models.append(model_2_accuracy_controls)

file_names_accuracy = ['models_0_accuracy', 'models_2_accuracy',
                       'models_0_controls_accuracy', 'models_2_controls_accuracy']
relevant_models_accuracy = [model_0_accuracy_models, model_2_accuracy_models,
                            model_0_accuracy_controls_models, model_2_accuracy_controls_models]

# os.chdir(base_directory + '/ols_s8_to_s17/')
# for fl in range(len(file_names_accuracy)):
#     file_name = file_names_accuracy[fl]
#     relevant_model = relevant_models_accuracy[fl]
#     with open(file_name + '_270423_Thurs.txt', 'w') as f:
#         f.write(summary_col(relevant_model, stars=True,
#                 float_format='%0.4f').as_latex())
#         f.close()

os.chdir(base_directory)

# %
# create ln belief/(1-belief)

df['belief_over_100'] = df['belief']/100
df['ln_belief'] = np.log(df['belief_over_100']/(1-df['belief_over_100']))
df['ln_prob'] = np.log(df['prob']/(1-df['prob']))

# replace values of ln_belief == inf/-inf with none

# %
df['ln_belief'].mask(df['ln_belief'] == float('inf'), None, inplace=True)
df['ln_belief'].mask(df['ln_belief'] == -float('inf'), None, inplace=True)

# recall that for var 2 and 4
# information was presented in the following way
# for values between 0 and 2, risk increases by blabla

# create new var2 and var4 which reflects that
df['var2_cat'] = df['var2']
df['var4_cat'] = df['var4']
df['var2_cat_more_noise'] = df['var2']
df['var4_cat_more_noise'] = df['var4']

df['var2_cat'].mask(df['var2'] <= 2, 1, inplace=True)
df['var4_cat'].mask(df['var4'] <= 2, 1, inplace=True)
df['var2_cat_more_noise'].mask(df['var2'] <= 1, float(0), inplace=True)
df['var4_cat_more_noise'].mask(df['var4'] <= 1, float(0), inplace=True)

for r in range(1, 6):
    r_plus_one = r+1
    two_r = r*2
    two_r_plus_one = two_r + 1
    two_r_plus_two = r_plus_one * 2
    df['var2_cat'].mask((df['var2'] > two_r) & (
        df['var2'] <= two_r_plus_two), r_plus_one, inplace=True)
    df['var4_cat'].mask((df['var4'] > two_r) & (
        df['var4'] <= two_r_plus_two), r_plus_one, inplace=True)

for j in range(1, 6):
    j_relevant = float(2*(j-1) + 1)
    j_plus_one = float(j_relevant + 1)
    j_plus_two = float(j_relevant + 2)
    df['var2_cat_more_noise'].mask((df['var2'] > j) & (
        df['var2'] <= j_plus_two), float(j_plus_one), inplace=True)
    df['var4_cat_more_noise'].mask((df['var4'] > r) & (
        df['var4'] <= j_plus_two), float(j_plus_one), inplace=True)

# %


# %
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

# %
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
# %
recovered_betas_sum_stats = pd.DataFrame(df.groupby(['treatment']).agg(
    {'coef_var1': ['mean', 'var'],
     'coef_var2': ['mean', 'var'],
     'coef_var3': ['mean', 'var'],
     'coef_var4': ['mean', 'var']}
))

recovered_betas_sum_stats

# %
recovered_betas_mean_dict = {}
for beta_name in participant_betas_name:
    recovered_betas_mean_dict_beta = {}
    for treat in treatment_names:
        recovered_betas_mean_dict_beta[treat] = recovered_betas_sum_stats[beta_name]['mean'][treat]
    recovered_betas_mean_dict[beta_name] = recovered_betas_mean_dict_beta


# %
beta_differences = pd.DataFrame(df.groupby(['treatment']).agg({'diff_var1': ['mean', 'var'],
                                                               'diff_var2': ['mean', 'var'],
                                                               'diff_var3': ['mean', 'var'],
                                                               'diff_var4': ['mean', 'var']}))
beta_differences

# add the beta differences to the recovered betas mean dict and
# convert to dataframe
diff_betas_names = ['diff_var1', 'diff_var2', 'diff_var3', 'diff_var4']

for diff_beta_name in diff_betas_names:
    diff_beta_mean_dict = {}
    for treat in treatment_names:
        diff_beta_mean_dict[treat] = beta_differences[diff_beta_name]['mean'][treat]
    recovered_betas_mean_dict[diff_beta_name] = diff_beta_mean_dict

pd.options.display.float_format = "{:,.4f}".format
recovered_betas_mean_df = pd.DataFrame(recovered_betas_mean_dict)

print(recovered_betas_mean_df.to_latex())
# %%
# models_coefficients = {}
# for i in range(4):
#     j = i+1
#     # how do different treatments affect the coefficient for variable 1,2,3,4?
#     model_var_x = ols('coef_var' + str(j) +
#                       ' ~ C(treatment) + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_char1 + timer_char2 + timer_char3 + timer_char4', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_' + str(j)] = model_var_x
#     # above + controls
#     model_var_x_controls = ols('coef_var' + str(j) +
#                                ' ~ C(treatment) + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_char1 + timer_char2 + timer_char3 + timer_char4 + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_controls_' + str(j)] = model_var_x_controls
#     # how do different treatments affect the difference between the participant beta and true beta for each variable?
#     model_var_diff_x = ols('diff_var' + str(j) +
#                            ' ~ C(treatment) + round_number + loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_char1 + timer_char2 + timer_char3 + timer_char4', data=df).fit(cov_type="HC0")
#     models_coefficients['diff_var_' + str(j)] = model_var_diff_x
#     # no intercept
#     model_var_diff_x_no_const = ols(
#         'diff_var' + str(j) + ' ~ C(treatment) - 1', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_diff_' +
#                         str(j) + '_no_const'] = model_var_diff_x_no_const
#     # how treatments affect diff between participant beta and true beta with controls
#     model_var_diff_x_controls = ols(
#         'diff_var' + str(j) + ' ~ C(treatment) + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_char1 + timer_char2 + timer_char3 + timer_char4+ age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_diff_' +
#                         str(j) + '_controls'] = model_var_diff_x_controls
#     # no intercept
#     model_var_diff_x_controls_no_const = ols(
#         'diff_var' + str(j) + ' ~ C(treatment) + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk -1', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_diff_' +
#                         str(j) + '_controls_no_const'] = model_var_diff_x_controls_no_const
#     # how different treatments affect difference with participant fixed effects
#     model_var_diff_x_part = ols(
#         'diff_var' + str(j) + ' ~ C(treatment) + C(label)', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_diff_' +
#                         str(j) + '_part'] = model_var_diff_x_part
#     # how treatments affect difference with fixed effects and controls
#     model_var_diff_x_controls_part = ols(
#         'diff_var' + str(j) + ' ~ C(treatment) + C(label) + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")
#     models_coefficients['model_var_diff_' +
#                         str(j) + '_controls_part'] = model_var_diff_x_controls_part

# # %

# # os.chdir(base_directory + '/ols_s8_to_s17/')
# # with open('beta_recovered_differences.txt', 'w') as f:
# #     f.write(summary_col([models_coefficients['diff_var_1'], models_coefficients['model_var_diff_1_controls'],
# #                          models_coefficients['diff_var_2'], models_coefficients['model_var_diff_2_controls'],
# #                          models_coefficients['diff_var_3'], models_coefficients['model_var_diff_3_controls'],
# #                          models_coefficients['diff_var_4'], models_coefficients['model_var_diff_4_controls'],], stars=True, float_format='%0.4f').as_latex())
# #     f.close()

# %

# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('beta_recovered_not_differences.txt', 'w') as f:
#     f.write(summary_col([models_coefficients['model_var_1'], models_coefficients['model_var_controls_1'],
#                          models_coefficients['model_var_2'], models_coefficients['model_var_controls_2'],
#                          models_coefficients['model_var_3'], models_coefficients['model_var_controls_3'],
#                          models_coefficients['model_var_4'], models_coefficients['model_var_controls_4'],], stars=True, float_format='%0.4f').as_latex())
#     f.close()

# %

# check whether those who didnt read are also those inconsistent in HL
df[['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4', 'timer_char7']
   ].loc[(df['HL_switchpoint'].isna()) | (df['HLLF_switchpoint'].isna())]

# probability of being inconsistent in HL given
# timers not read

# create variable for inconsistent HL
df['inconsistent_HL'] = df['HL_switchpoint'].isna()
df['inconsistent_HLLF'] = df['HLLF_switchpoint'].isna()

# %
df['inconsistent_HL'].mask(df['inconsistent_HL'] == True, 1, inplace=True)
df['inconsistent_HL'].mask(df['inconsistent_HL'] == False, 0, inplace=True)
df['inconsistent_HLLF'].mask(df['inconsistent_HLLF'] == True, 1, inplace=True)
df['inconsistent_HLLF'].mask(df['inconsistent_HLLF'] == False, 0, inplace=True)
# %
df['inconsistent_HL'] = pd.to_numeric(df['inconsistent_HL'])
df['inconsistent_HLLF'] = pd.to_numeric(df['inconsistent_HLLF'])
timer_HL = probit(
    'inconsistent_HL ~ timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7', data=df).fit(cov_type="HC0")
timer_HL.summary()
# spending more time on char1 is associated with higher likelihood
# of individual being inconsistent in HL
# spending more time on char2 is associated with lower likelihood
# of individual being inconsistent in HL
# this is why i want to see the differences spent
# between each timer are significant
# %
timer_HLLF = probit(
    'inconsistent_HLLF ~ timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7', data=df).fit(cov_type="HC0")
timer_HLLF.summary()
# nothing significant
# %
# do people spend more time on first two timers compared to latter two

# Mann Whitney U test between coefficients
timer_names = ['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4']

mannwhitneyu_timers_results = {}
for treat in treatment_names:
    df_one_treatment = df.loc[df['treatment'] == treat]
    timer_in_question = {}
    for timer in timer_names:
        timer_stat_p = {}
        other_timers = []
        for ot in timer_names:
            if ot != timer:
                other_timers.append(ot)
        for other_timer in other_timers:
            stat_t, p_t = mannwhitneyu(
                df_one_treatment[timer], df_one_treatment[other_timer])
            timer_stat_p[other_timer] = [stat_t, p_t]
        timer_in_question[timer] = timer_stat_p
    mannwhitneyu_timers_results[treat] = timer_in_question
# %
# only significant differences between (3,1) (4,1) (3,2) (4,2)
df['timer_diff_3_1'] = df['timer_char3'] - df['timer_char1']
df['timer_diff_4_1'] = df['timer_char4'] - df['timer_char1']
df['timer_diff_3_2'] = df['timer_char3'] - df['timer_char2']
df['timer_diff_4_2'] = df['timer_char4'] - df['timer_char2']

timer_diffs = pd.DataFrame(df.groupby(['treatment']).agg({'timer_diff_3_1': 'mean',
                                                          'timer_diff_4_1': 'mean',
                                                          'timer_diff_3_2': 'mean',
                                                          'timer_diff_4_2': 'mean', })).reset_index()

timer_diffs.to_latex()
# in baseline, fullinfo, neginfo and varinfo
# time spent on 1 and 2 exceeds time spent on 3 and 4

# for posinfo, only (4,1) and (4,2) significant (but even if more were significant)
# time spent on 4 exceeds time spent on 1 and 2

# %

var3_sum_stats = pd.DataFrame(df.groupby(['var3', 'treatment']).agg(
    {'coef_var3': ['mean', 'var'],
     'diff_var3': ['mean', 'var']}))

var3_sum_stats
# %
var1_sum_stats = pd.DataFrame(df.groupby(['var1', 'treatment']).agg(
    {'coef_var1': ['mean', 'var'],
     'diff_var1': ['mean', 'var']}))

var1_sum_stats
# %
var2_sum_stats = pd.DataFrame(df.groupby(['var2', 'treatment']).agg(
    {'coef_var2': ['mean', 'var'],
     'diff_var2': ['mean', 'var']}))

var2_sum_stats[0:30]

# %
var4_sum_stats = pd.DataFrame(df.groupby(['var4']).agg(
    {'coef_var4': ['mean', 'var'],
     'diff_var4': ['mean', 'var']}))

var4_sum_stats
# %

# WILLINGNESS TO PAY HISTOGRAM

df['WTP'].hist(by=df['loss_amount'])
# %
# mann whitney U to see whether WTP differs across treatments for certain losses

mannwhitneyu_wtp_results = {}

for loss in [20, 40, 60, 80, 100]:
    mannwhitneyu_wtp_result_treat = {}
    for treat in treatment_names:
        other_treatments = []
        treat_stat_p = {}
        for ot in treatment_names:
            if ot != treat:
                other_treatments.append(ot)
        for other_treat in other_treatments:
            stat_t, p_t = mannwhitneyu(df['WTP'].loc[(df['loss_amount'] == loss) & (
                df['treatment'] == treat)], df['WTP'].loc[(df['loss_amount'] == loss) & (df['treatment'] == other_treat)])
            treat_stat_p[other_treat] = [stat_t, p_t]
        mannwhitneyu_wtp_result_treat[treat] = treat_stat_p
    mannwhitneyu_wtp_results[loss] = mannwhitneyu_wtp_result_treat

mannwhitneyu_wtp_results

# loss 20: sig diff btw baseline and neginfo
# loss 80: sig diff btw fullinfo and neginfo, neginfo and allinfo
# loss 100: sig diff btw neginfo and allinfo

# %
# what if we calculated WTP as percentage of loss amount
# mathematically it's the same

# %

wtp_per_loss = pd.DataFrame(df.groupby(['treatment', 'loss_amount']).agg(
    {'WTP': ['mean', 'var']}))

wtp_all_loss_dict = {}
for treat in treatment_names:
    wtp_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        wtp_for_treat_dict[loss] = wtp_per_loss['WTP']['mean'][treat][loss]
    wtp_all_loss_dict[treat] = wtp_for_treat_dict

wtp_all_loss_df = pd.DataFrame(wtp_all_loss_dict)
# plt.bar(wtp_for_loss, )

wtp_all_loss_df['loss_amount'] = wtp_all_loss_df.index

wtp_all_loss_df


# %
# pivot table
wtp_mean_losses = pd.pivot_table(pd.DataFrame(wtp_per_loss['WTP']['mean']), index=[
                                 'loss_amount'], columns=['treatment'])

# convert from hierarchical df to normal df
wtp_mean_losses.columns = wtp_mean_losses.columns.get_level_values(1)

print(wtp_mean_losses.to_latex())
# %
colors_for_wtp = ['darkgray', 'black', 'red', 'green', 'blue']
loss_amounts = [20, 40, 60, 80, 100]
ind = np.arange(5)
width = 0.15
treatment_names_for_display = ['Baseline',
                               'Full info', 'Neg info', 'Pos info', 'All info']
wtp_bars = []
for l in range(len(treatment_names)):
    wtp_bars.append(plt.bar(
        ind+width*l, wtp_all_loss_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, loss_amounts)

os.chdir(base_directory)
plt.xlabel('Loss amounts')
plt.ylabel('Mean willingness to pay')
plt.legend(tuple(wtp_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/mean_wtp_by_treat_and_loss.png')
plt.show()

# %
# create insurance take up binary variable
# df['insurance_takeup'] = ~df['WTP'].eq(0)
df = df.assign(insurance_takeup=[0 for i in range(len(df))])

# %
df['insurance_takeup'].mask((df['WTP'] > df['insurance']), 1, inplace=True)

df[['insurance_takeup', 'WTP', 'insurance']]
# %
# #
# df['insurance_takeup'].mask(df['insurance_takeup'] == True, 1, inplace=True)
# df['insurance_takeup'].mask(df['insurance_takeup'] == False, 0, inplace=True)

df['insurance_takeup'] = pd.to_numeric(df['insurance_takeup'])

insurance_take_up_rate = pd.DataFrame(df.groupby(['treatment', 'loss_amount']).agg(
    {'insurance_takeup': ['mean']}))  # mean because i want take up rate
# which is sum over count

insurance_take_up_rate
# %
takeup_all_loss_dict = {}
for treat in treatment_names:
    takeup_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        takeup_for_treat_dict[loss] = insurance_take_up_rate['insurance_takeup']['mean'][treat][loss]
    takeup_all_loss_dict[treat] = takeup_for_treat_dict

takeup_all_loss_df = pd.DataFrame(takeup_all_loss_dict)
# plt.bar(wtp_for_loss, )

# takeup_all_loss_df['loss_amount'] = takeup_all_loss_df.index

takeup_all_loss_df

print(takeup_all_loss_df.to_latex())
# %
takeup_bars = []
for l in range(len(treatment_names)):
    takeup_bars.append(plt.bar(
        ind+width*l, takeup_all_loss_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, loss_amounts)

os.chdir(base_directory)
plt.xlabel('Loss amounts')
plt.ylabel('Insurance take-up rate')
# plt.legend(tuple(takeup_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/insurance_takeup_rate_by_treat_and_loss.png')
plt.show()
# %

mannwhitneyu_takeup_rate_loss_results = {}

for loss in [20, 40, 60, 80, 100]:
    mannwhitneyu_takeup_rate_loss_treat_results = {}
    for treat in treatment_names:
        other_treatments = []
        treat_stat_p = {}
        for ot in treatment_names:
            if ot != treat:
                other_treatments.append(ot)
        for other_treat in other_treatments:
            stat_t, p_t = mannwhitneyu(df['insurance_takeup'].loc[(df['loss_amount'] == loss) & (
                df['treatment'] == treat)], df['insurance_takeup'].loc[(df['loss_amount'] == loss) & (df['treatment'] == other_treat)])
            treat_stat_p[other_treat] = [stat_t, p_t]
        mannwhitneyu_takeup_rate_loss_treat_results[treat] = treat_stat_p
    mannwhitneyu_takeup_rate_loss_results[loss] = mannwhitneyu_takeup_rate_loss_treat_results

mannwhitneyu_takeup_rate_loss_results

# insurance take up
# loss of 40: sig diff fullinfo and neginfo
# loss of 60: sig diff baseline and allinfo, fullinfo and allinfo
# loss of 80: sig diff baseline and allinfo, neginfo and allinfo

# WTP
# loss 20: sig diff btw baseline and neginfo
# loss 80: sig diff btw fullinfo and neginfo, neginfo and allinfo
# loss 100: sig diff btw neginfo and allinfo

# %
# should WTP control for losses in previous rounds?
df = df.assign(state_lag_1=[None for i in range(len(df))])

# %
# convert round to date time format
df['round_date_format'] = df['round_number']
df['round_date_format'] = pd.to_datetime(df['round_date_format'])
# # create a lagged variable of state
participants = df['label'].unique()

for participant in participants:
    df['state_lag_1'].loc[df['label'] ==
                          participant] = df['state'].loc[df['label'] == participant].shift(1)


# %
df = df.assign(loss_experience=[None for i in range(len(df))])

for participant in participants:
    df['loss_experience'].loc[(df['label'] ==
                               participant) & (df['WTP'] >= df['insurance'])] = 0
    df['loss_experience'].loc[(df['label'] ==
                               participant) & (df['WTP'] < df['insurance']) & (df['state'] == 'healthy')] = 0
    df['loss_experience'].loc[(df['label'] ==
                               participant) & (df['WTP'] < df['insurance']) & (df['state'] == 'ill')] = 1
    # df['state'].loc[df['label'] == participant].shift(1)
# %
df = df.assign(loss_experience_lag_1=[None for i in range(len(df))])

for participant in participants:
    df['loss_experience_lag_1'].loc[df['label'] ==
                                    participant] = df['loss_experience'].loc[df['label'] == participant].shift(1)

# %

df['loss_experience_lag_1'] = pd.to_numeric(df['loss_experience_lag_1'])
# %
df['state_lag_1'].mask(df['state_lag_1'] == 'healthy', 0, inplace=True)
df['state_lag_1'].mask(df['state_lag_1'] == 'ill', 1, inplace=True)
# %

# %
takeup_insurance_prob = probit(
    'insurance_takeup ~ treatment + prob +  number_entered + HL_switchpoint + loss_amount + round_number + timer_all_chars', data=df).fit(cov_type='HC0')

takeup_insurance_prob_belief = probit(
    'insurance_takeup ~ treatment + belief + prob + number_entered + HL_switchpoint + loss_amount + round_number + timer_all_chars', data=df).fit(cov_type='HC0')

takeup_insurance_var = probit(
    'insurance_takeup ~ treatment + number_entered + HL_switchpoint + loss_amount + round_number + timer_all_chars + var1 + var2 + var3 + var4', data=df).fit(cov_type='HC0')

takeup_insurance_var_belief = probit(
    'insurance_takeup ~ treatment + belief  +  number_entered + HL_switchpoint + loss_amount + round_number + timer_all_chars + var1 + var2 + var3 + var4', data=df).fit(cov_type='HC0')

# %
takeup_insurance_controls_prob = probit(
    'insurance_takeup ~ treatment + belief + prob + loss_amount  + round_number + timer_all_chars + age + gender + smoker + number_entered +HL_switchpoint+ health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

takeup_insurance_controls_var = probit(
    'insurance_takeup ~ treatment + belief + var1 + var2 + var3 + var4 + loss_amount + round_number + timer_all_chars  + age + gender + smoker + number_entered + HL_switchpoint + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')


# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('takeup_rate.txt', 'w') as f:
#     f.write(summary_col([takeup_insurance_prob,
#                          takeup_insurance_prob_belief,
#                          takeup_insurance_controls_prob,
#                          takeup_insurance_var,
#                          takeup_insurance_var_belief,
#                          takeup_insurance_controls_var], stars=True, float_format='%0.4f').as_latex())
#     f.close()

# %
# create a version of number_entered based on unique values of number_entered
# get unique number_entered
unique_number_entered = df['number_entered'].unique().tolist()
unique_number_entered.sort(reverse=True)  # least risk averse first
unique_number_entered

# for each unique value assign rank.
# for those who entered 50, rank = 1 and those who entered 0, rank = len(unique_number_entered)

unique_number_entered_rank = {
    unique_number_entered[k]: k+1 for k in range(len(unique_number_entered))}
unique_number_entered_rank

# %
df = df.assign(risk_rank=[None for i in range(len(df))])

for une in unique_number_entered:
    df['risk_rank'].loc[df['number_entered'] ==
                        une] = unique_number_entered_rank[une]

df['risk_rank'] = pd.to_numeric(df['risk_rank'])

df[['number_entered', 'risk_rank']]

# %
# higher HL_switchpoint = more risk averse
# %
#######
# instead of number entered, use ranking of risk aversion
# as rank increases, risk aversion increases
#


# %

# number_entered_risk = df[['number_entered','risk','riskdriving','riskfinance','risksport','riskjob','riskhealth']].corr()
# number_entered_risk

# import seaborn as sn
# sn.heatmap(number_entered_risk, annot=True, cmap = sn.cm.rocket_r)
# plt.show()
number_entered_risk = ols(
    'number_entered ~ risk + riskdriving + riskfinance + risksport + riskjob + riskhealth', data=df).fit(cov_type='HC0')
number_entered_risk.summary()

# as risk taking in finance increases, number entered increases
# %
# aversion_risk = df[['aversion_rank','risk','riskdriving','riskfinance','risksport','riskjob','riskhealth']].corr()
# aversion_risk
# sn.heatmap(aversion_risk, annot=True, cmap = sn.cm.rocket_r)
# plt.show()
# aversion_risk = ols(
#     'aversion_rank ~ risk + riskdriving + riskfinance + risksport + riskjob + riskhealth', data=df).fit(cov_type='HC0')
# aversion_risk.summary()

# as risk taking increases, aversion rank decreases (lowest rank, most risk taking)

# # %
# # # %
# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('risk_measure_correlation.txt', 'w') as f:
#     f.write(summary_col([number_entered_risk, aversion_risk], stars=True, float_format='%0.4f').as_latex())
#     f.close()

# %
# %
# calculate welfare

# find price = avg probability * loss amoouont
prob_avg = pd.DataFrame(df.groupby(
    ['treatment', 'loss_amount']).agg({'prob': 'mean'}))
prob_avg
# %
df = df.assign(fair_price=[None for i in range(len(df))])

for treat in treatment_names:
    for loss in [20, 40, 60, 80, 100]:
        df['fair_price'].loc[(df['loss_amount'] == loss) & (
            df['treatment'] == treat)] = prob_avg['prob'][treat][loss]*loss

# %
df['consumer_surplus'] = df['WTP']-df['fair_price']

df[['consumer_surplus', 'WTP', 'fair_price']]
# %
df[['WTP']].sort_values(by=['WTP'], ascending=False).values.tolist()

df[['WTP']].loc[df['loss_amount'] == 20].sort_values(
    by=['WTP'], ascending=False).values.tolist()

wtp_for_plot = {}
for loss in [20, 40, 60, 80, 100]:
    wtp_for_loss = df[['WTP']].loc[df['loss_amount'] == loss].sort_values(
        by=['WTP'], ascending=False).values.tolist()
    wtp_for_plot_loss = []
    qty_for_plot_loss = []
    count = 1
    for item in wtp_for_loss:
        wtp_for_plot_loss.append(item[0])
        qty_for_plot_loss.append(count)
        count += 1
    temp_df_for_loss = pd.DataFrame(
        list(zip(wtp_for_plot_loss, qty_for_plot_loss)), columns=['wtp', 'qty'])
    wtp_for_plot[loss] = temp_df_for_loss

# %
# colors = ['darkgray', 'black', 'red', 'green', 'blue']
# df['fair_price'].loc[(df['loss_amount'] == 20) & (df['treatment'] == 'baseline')]

# initialize subplot function
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
loss_amounts = [20, 40, 60, 80, 100]
titles = ['Loss = 20', 'Loss = 40', 'Loss = 60', 'Loss = 80', 'Loss = 100']

# %
for i in range(len(loss_amounts)):
    loss = loss_amounts[i]
    axis[axes[i][0], axes[i][1]].plot(
        wtp_for_plot[loss]['qty'], wtp_for_plot[loss]['wtp'], linestyle='-')
    axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
    axis[axes[i][0], axes[i][1]].axhline(0.5*loss, color='red')
    axis[axes[i][0], axes[i][1]].xaxis.set_tick_params(labelbottom=False)
    axis[axes[i][0], axes[i][1]].set_xticks([])
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('WTP')

os.chdir(base_directory)
figure.delaxes(axis[2, 1])
# plt.savefig('plots_s8_to_s17/WTP_consumer_surplus.png')
plt.show()


# there are minor variations in the fair price for different treatments,
# it is hard to discern in the figure
# hence use probability of 0.5 in the figure
# and calculate actual surplus with true fair price
#
# %

# calculate the total consumer surplus for each treatment
consumer_surplus_treatment = pd.DataFrame(df.loc[df['consumer_surplus'] >= 0].groupby(
    ['treatment', 'loss_amount']).agg({'consumer_surplus': ['sum', 'mean']}))

consumer_surplus_treatment

# %
fair_prices = pd.DataFrame(df.groupby(
    ['treatment', 'loss_amount']).agg({'fair_price': ['mean']}))

fair_prices
#
# %
fair_prices_dict = {}
consumer_surplus_dict = {}
for treat in treatment_names:
    fair_prices_for_treat_dict = {}
    consumer_surplus_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        fair_prices_for_treat_dict[loss] = fair_prices['fair_price']['mean'][treat][loss]
        consumer_surplus_for_treat_dict[loss] = consumer_surplus_treatment['consumer_surplus']['sum'][treat][loss]
    fair_prices_dict[treat] = fair_prices_for_treat_dict
    consumer_surplus_dict[treat] = consumer_surplus_for_treat_dict

fair_prices_df = pd.DataFrame(fair_prices_dict)
consumer_surplus_df = pd.DataFrame(consumer_surplus_dict)

print(fair_prices_df.to_latex())
print(consumer_surplus_df.to_latex())
# %
df[['consumer_surplus', 'fair_price']] = df[['consumer_surplus',
                                             'fair_price']].apply(pd.to_numeric, errors='coerce')
# %
surplus_basic = ols(
    'consumer_surplus ~ treatment + belief + loss_amount + loss_experience_lag_1 + number_entered + round_number + timer_all_chars + var1 + var2 + var3 + var4', data=df).fit(cov_type='HC0')
surplus_basic.summary()
# %
# %

# wtp with controls
surplus_controls = ols(
    'consumer_surplus ~ treatment + belief + loss_amount + loss_experience_lag_1 + number_entered + loss_experience_lag_1 + round_number + timer_all_chars + var1 + var2 + var3 + var4 + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')
surplus_controls.summary()
# %
# %

# over and underinsurance

# what counts as overinsurance
# purchasing too much coverage

# what counts as underinsurance
# purchasing too little coverage

# in the data no one has a consumer surplus of zero
# so either over or underinsured
df = df.assign(overinsure=[None for i in range(len(df))])
df['overinsure'].mask(df['consumer_surplus'] >= 0, 1, inplace=True)
df['overinsure'].mask(df['consumer_surplus'] < 0, 0, inplace=True)

# %
# calculate the total consumer surplus for each treatment
over_under_insurance = pd.DataFrame(df.groupby(
    ['loss_amount', 'overinsure', 'treatment']).agg({'consumer_surplus': ['mean', 'var', 'count'],
                                                     'fair_price': ['mean', 'var', 'count']}))

over_under_insurance


# %
overinsure_dict = {}
underinsure_dict = {}
for treat in treatment_names:
    overinsure_for_treat_dict = {}
    underinsure_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        overinsure_for_treat_dict[loss] = over_under_insurance['consumer_surplus']['mean'][loss][1][treat] / \
            over_under_insurance['fair_price']['mean'][loss][1][treat]
        underinsure_for_treat_dict[loss] = over_under_insurance['consumer_surplus']['mean'][loss][0][treat] / \
            over_under_insurance['fair_price']['mean'][loss][0][treat]
    overinsure_dict[treat] = overinsure_for_treat_dict
    underinsure_dict[treat] = underinsure_for_treat_dict

overinsure_df = pd.DataFrame(overinsure_dict)
underinsure_df = pd.DataFrame(underinsure_dict)

over_under_insurance
overinsure_df
underinsure_df

# %
ind = np.arange(26.25, 111.25, 20)
width = 3.5

#
for l in range(len(treatment_names)):
    for ls in range(len(loss_amounts)):
        plt.barh(ind[ls]-width*l, overinsure_df[treatment_names[l]]
                 [loss_amounts[ls]], width, color=colors_for_wtp[l])
        plt.barh(ind[ls]-width*l, underinsure_df[treatment_names[l]]
                 [loss_amounts[ls]], width, color=colors_for_wtp[l])
        plt.axvline(x=0, linestyle='--', lw=0.7)

# use same legend as before
ax = plt.gca()
ax.tick_params(axis='y', which='both', length=0)
# plt.ylabel('Loss amounts')
plt.xlabel('Diff WTP & actuarially fair price, %')
# plt.legend(tuple(wtp_bars), tuple(treatment_names_for_display))
os.chdir(base_directory)
# plt.savefig('plots_s8_to_s17/over_and_under_insurance_percent.png')
plt.show()

# do percentages

# %
df['consumer_surplus_belief'] = df['WTP'] - \
    df['belief_over_100']*df['loss_amount']

# %
# find average "fair" price for each loss amount based on beliefs

# find price = avg belief * loss amoouont
belief_avg = pd.DataFrame(df.groupby(
    ['treatment', 'loss_amount']).agg({'belief_over_100': 'mean'}))

df = df.assign(fair_price_belief=[None for i in range(len(df))])

for treat in treatment_names:
    for loss in [20, 40, 60, 80, 100]:
        df['fair_price_belief'].loc[(df['loss_amount'] == loss) & (
            df['treatment'] == treat)] = belief_avg['belief_over_100'][treat][loss]*loss

# no point plotting demand curves since average probabilities are about the same as average
# beliefs in each treatment
# %

# in the data no one has a consumer surplus of zero
# so either over or underinsured
df = df.assign(overinsure_belief=[None for i in range(len(df))])
df['overinsure_belief'].mask(
    df['consumer_surplus_belief'] >= 0, 1, inplace=True)
df['overinsure_belief'].mask(
    df['consumer_surplus_belief'] < 0, 0, inplace=True)
# do the over and underinsurance bar chart plot but
# for beliefs instead of actuarially fair price

over_under_insurance_belief = pd.DataFrame(df.groupby(
    ['loss_amount', 'overinsure_belief', 'treatment']).agg({'consumer_surplus_belief': ['mean', 'var', 'count'],
                                                            'fair_price_belief': ['mean', 'var', 'count']}))

over_under_insurance_belief


# %
overinsure_belief_dict = {}
underinsure_belief_dict = {}
for treat in treatment_names:
    overinsure_belief_for_treat_dict = {}
    underinsure_belief_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        overinsure_belief_for_treat_dict[loss] = over_under_insurance_belief['consumer_surplus_belief'][
            'mean'][loss][1][treat]/over_under_insurance_belief['fair_price_belief']['mean'][loss][1][treat]
        underinsure_belief_for_treat_dict[loss] = over_under_insurance_belief['consumer_surplus_belief'][
            'mean'][loss][0][treat]/over_under_insurance_belief['fair_price_belief']['mean'][loss][0][treat]
    overinsure_belief_dict[treat] = overinsure_belief_for_treat_dict
    underinsure_belief_dict[treat] = underinsure_belief_for_treat_dict

overinsure_belief_df = pd.DataFrame(overinsure_belief_dict)
underinsure_belief_df = pd.DataFrame(underinsure_belief_dict)

over_under_insurance_belief
overinsure_belief_df
underinsure_belief_df

# %
ind = np.arange(26.25, 111.25, 20)
width = 3.5

#
for l in range(len(treatment_names)):
    for ls in range(len(loss_amounts)):
        plt.barh(ind[ls]-width*l, overinsure_belief_df[treatment_names[l]]
                 [loss_amounts[ls]], width, color=colors_for_wtp[l])
        plt.barh(ind[ls]-width*l, underinsure_belief_df[treatment_names[l]]
                 [loss_amounts[ls]], width, color=colors_for_wtp[l])
        plt.axvline(x=0, linestyle='--', lw=0.7)

# use same legend as before
ax = plt.gca()
ax.tick_params(axis='y', which='both', length=0)
# plt.ylabel('Loss amounts')
plt.xlabel('Diff WTP & belief fair price, %')
# plt.legend(tuple(wtp_bars), tuple(treatment_names_for_display))
os.chdir(base_directory)
# plt.savefig('plots_s8_to_s17/over_and_under_insurance_belief_percent.png')
plt.show()

# %
# plot consumer surplus in each treatment

# consumer surplus mean for treatment
consumer_surplus_mean_treat = {}
for treat in treatment_names:
    consumer_surplus_mean_treat[treat] = sum(consumer_surplus_df[treat])

# %
# add row
consumer_surplus_df = consumer_surplus_df.append(pd.DataFrame(
    [consumer_surplus_mean_treat], index=[120], columns=consumer_surplus_df.columns))
# %
ind = np.arange(6)
width = 0.15
new_loss_amounts = [20, 40, 60, 80, 100, 'Mean']
surplus_bars = []

for l in range(len(treatment_names)):
    surplus_bars.append(plt.bar(
        ind+width*l, consumer_surplus_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, new_loss_amounts)

os.chdir(base_directory)
plt.axvline(x=4.775, linestyle='--', lw=0.7)
plt.xlabel('Loss amounts')
plt.ylabel('Consumer surplus, actuarially fair price')
plt.legend(tuple(surplus_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17\\consumer_surplus_fair_price.png')
plt.show()
# %

# calculate the total consumer surplus for each treatment
consumer_surplus_belief_treatment = pd.DataFrame(df.loc[df['consumer_surplus_belief'] >= 0].groupby(
    ['treatment', 'loss_amount']).agg({'consumer_surplus_belief': ['sum', 'mean']}))

consumer_surplus_belief_treatment

# %
fair_prices_belief = pd.DataFrame(df.groupby(
    ['treatment', 'loss_amount']).agg({'fair_price_belief': ['mean']}))

fair_prices_belief
# %
fair_prices_belief_dict = {}
consumer_surplus_belief_dict = {}
for treat in treatment_names:
    fair_prices_belief_for_treat_dict = {}
    consumer_surplus_belief_for_treat_dict = {}
    for loss in [20, 40, 60, 80, 100]:
        fair_prices_belief_for_treat_dict[loss] = fair_prices_belief['fair_price_belief']['mean'][treat][loss]
        consumer_surplus_belief_for_treat_dict[loss] = consumer_surplus_belief_treatment[
            'consumer_surplus_belief']['sum'][treat][loss]
    fair_prices_belief_dict[treat] = fair_prices_belief_for_treat_dict
    consumer_surplus_belief_dict[treat] = consumer_surplus_belief_for_treat_dict

consumer_surplus_belief_df = pd.DataFrame(consumer_surplus_belief_dict)
# %
consumer_surplus_belief_mean_treat = {}

for treat in treatment_names:
    consumer_surplus_belief_mean_treat[treat] = sum(
        consumer_surplus_belief_df[treat])

# %
# add row
consumer_surplus_belief_df = consumer_surplus_belief_df.append(pd.DataFrame(
    [consumer_surplus_belief_mean_treat], index=[120], columns=consumer_surplus_df.columns))
# %
ind = np.arange(6)
width = 0.15
new_loss_amounts = [20, 40, 60, 80, 100, 'Mean']
surplus_belief_bars = []

for l in range(len(treatment_names)):
    surplus_belief_bars.append(plt.bar(
        ind+width*l, consumer_surplus_belief_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, new_loss_amounts)

os.chdir(base_directory)
plt.axvline(x=4.775, linestyle='--', lw=0.7)
ax = plt.gca()
ax.set_ylim([0, 2000])
plt.xlabel('Loss amounts')
plt.ylabel('Consumer surplus, belief price')
# plt.legend(tuple(surplus_belief_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17\\consumer_surplus_belief_price.png')
plt.show()


# %

# 20, takeup: neg and full, wtp: neg and base
# 40, takeup: neg and pos, pos and all, wtp: nothing
# 60, takeup: base and pos, base and all, wtp: nothing
# 80, takeup: base and neg, base and pos, wtp: full and neg
# 100, takeup: nothing, wtp: neg and all
loss_amounts = [20, 40, 60, 80, 100]
ind = np.arange(5)
width = 1
wtp_dots = []

# %
for l in range(len(treatment_names)):
    wtp_dots.append(plt.plot(df['loss_amount'].loc[df['treatment'] == treatment_names[l]]+l+width*l, df['WTP'].loc[df['treatment'] == treatment_names[l]],
                    markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l]))  # , label=treatment_names[l]))
    plt.xticks(
        [loss_amounts[m]+4 for m in range(len(loss_amounts))], loss_amounts)

os.chdir(base_directory)
plt.xlabel('Loss amounts')
plt.ylabel('Willingness to pay')
plt.legend()  # tuple(wtp_dots), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_scatter_by_loss_and_treat.png')
plt.show()

# %%
# initialize subplot function
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
loss_amounts = [20, 40, 60, 80, 100]
titles = ['Loss = 20', 'Loss = 40', 'Loss = 60', 'Loss = 80', 'Loss = 100']

# %
for i in range(len(loss_amounts)):
    loss = loss_amounts[i]
    for l in range(len(treatment_names)):
        # axis[axes[i][0], axes[i][1]].plot(df['prob'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
        a, b = np.polyfit(df['prob'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], 1)
        axis[axes[i][0], axes[i][1]].plot(df['prob'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], a*df['prob'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)]+b, color=colors[l], lw=0.7)
        axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
        axis[axes[i][0], axes[i][1]].axhline(0.5*loss, linestyle='--', lw=0.5)
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('Willingness to pay')
    if i == 4:
        axis[axes[i][0], axes[i][1]].set_xlabel('True probability')
        axis[axes[i][0], axes[i][1]].legend(bbox_to_anchor=(2, 1))
    if (i != 4) & (i != 3):
        axis[axes[i][0], axes[i][1]].set_xticks([])

figure.delaxes(axis[2, 1])
os.chdir(base_directory)
# plt.savefig('plots_s8_to_s17/wtp_per_prob_loss_treat.png')
plt.show()


# %%
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
loss_amounts = [20, 40, 60, 80, 100]
titles = ['Loss = 20', 'Loss = 40', 'Loss = 60', 'Loss = 80', 'Loss = 100']

# %
for i in range(len(loss_amounts)):
    loss = loss_amounts[i]
    for l in range(len(treatment_names)):
        # axis[axes[i][0], axes[i][1]].plot(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
        a, b = np.polyfit(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], 1)
        axis[axes[i][0], axes[i][1]].plot(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], a*df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)]+b, color=colors[l], lw=0.7, label=treatment_names_for_display[l])
        axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
        axis[axes[i][0], axes[i][1]].axhline(
            [0.5*loss], linestyle='--', lw=0.5)
        # axis[axes[i][0], axes[i][1]].axline(, linestyle='--', lw=0.5)
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('Willingness to pay')
    if i == 4:
        axis[axes[i][0], axes[i][1]].set_xlabel('Beliefs')
        axis[axes[i][0], axes[i][1]].legend(bbox_to_anchor=(2, 1))
    if (i != 4) & (i != 3):
        axis[axes[i][0], axes[i][1]].set_xticks([])

axis[2, 1].legend()
figure.delaxes(axis[2, 1])
os.chdir(base_directory)
# plt.legend()
# plt.savefig('plots_s8_to_s17/wtp_belief_treat_loss.png')
plt.show()

# %%

for loss in loss_amounts:
    for l in range(len(treatment_names)):
        plt.plot(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
        a, b = np.polyfit(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], 1)
        plt.plot(df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (
            df['loss_amount'] == loss)], a*df['belief_over_100'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)]+b, color=colors[l], lw=0.7)
        plt.title('Loss amount = ' + str(loss))
        plt.axhline(0.5*loss, linestyle='--', lw=0.5)
    plt.show()

# %
# plot belief vs probability


for l in range(len(treatment_names)):
    wtp_dots.append(plt.plot(df['prob'].loc[df['treatment'] == treatment_names[l]], df['belief_over_100'].loc[df['treatment'] == treatment_names[l]],
                    markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l]))  # , label=treatment_names[l]))
    a, b = np.polyfit(df['prob'].loc[df['treatment'] == treatment_names[l]],
                      df['belief_over_100'].loc[df['treatment'] == treatment_names[l]], 1)
    plt.plot(df['prob'].loc[(df['treatment'] == treatment_names[l])], a *
             df['prob'].loc[(df['treatment'] == treatment_names[l])]+b, color=colors[l], lw=0.7)
    # plt.xticks([loss_amounts[m]+4 for m in range(len(loss_amounts))], loss_amounts)

os.chdir(base_directory)
plt.xlabel('Probability')
plt.ylabel('Belief')
plt.legend()  # tuple(wtp_dots), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/belief_prob_scatter_fit.png')
plt.show()

# %
# same as scatter plot from above but instead of
# using all beliefs, use mean to make it look neater

belief_by_prob_treat = pd.DataFrame(df.groupby(['treatment', 'prob']).agg({
                                    'belief_over_100': 'mean'})).reset_index()

# get unique probabilities
unique_probs = df['prob'].unique().tolist()
unique_probs.sort()

belief_by_prob_treat

treatment_names_for_display = ['Baseline',
                               'Full info', 'Neg info', 'Pos info', 'All info']

for l in range(len(treatment_names)):
    plt.plot(belief_by_prob_treat['prob'].loc[belief_by_prob_treat['treatment'] == treatment_names[l]], belief_by_prob_treat['belief_over_100'].loc[belief_by_prob_treat['treatment']
             == treatment_names[l]], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
    a, b = np.polyfit(df['prob'].loc[df['treatment'] == treatment_names[l]],
                      df['belief_over_100'].loc[df['treatment'] == treatment_names[l]], 1)
    plt.plot(df['prob'].loc[(df['treatment'] == treatment_names[l])], a *
             df['prob'].loc[(df['treatment'] == treatment_names[l])]+b, color=colors[l], lw=0.7)
    # plt.xticks(ind+width+0.17, ['%.4f' % elem for elem in unique_probs], rotation=90)

os.chdir(base_directory)
plt.xlabel('Probability')
plt.ylabel('Mean belief')
plt.legend()
# plt.savefig('plots_s8_to_s17/belief_prob_mean_fit.png')
plt.show()
# # %%

# belief_by_prob_treat_df = np.round(pd.pivot_table(belief_by_prob_treat,
#                                                   values='belief_over_100',
#                                                   index='prob',
#                                                   columns='treatment',
#                                                   fill_value=0), 4)

# belief_by_prob_treat_df

# # %%
# ind = np.arange(len(unique_probs))
# width = 0.17
# treatment_names_for_display = ['Baseline',
#                                'Full info', 'Neg info', 'Pos info', 'All info']
# belief_prob_bars = []
# for l in range(len(treatment_names)):
#     plt.plot(ind+width, belief_by_prob_treat_df[treatment_names[l]], markers[l],
#              markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
#     plt.xticks(ind+width+0.17, ['%.4f' %
#                elem for elem in unique_probs], rotation=90)

# os.chdir(base_directory)
# plt.xlabel('Probability')
# plt.ylabel('Mean belief')
# plt.legend(tuple(belief_prob_bars), tuple(treatment_names_for_display))
# # plt.savefig('plots_s8_to_s17/belief_treat_prob.png')
# plt.show()

# create a version of number_entered based on unique values of number_entered
# get unique number_entered
unique_number_entered = df['number_entered'].unique().tolist()
unique_number_entered.sort(reverse=True)  # least risk averse first
unique_number_entered

# for each unique value assign rank.
# for those who entered 50, rank = 1 and those who entered 0, rank = len(unique_number_entered)

unique_number_entered_rank = {
    unique_number_entered[k]: k+1 for k in range(len(unique_number_entered))}
unique_number_entered_rank

# %
df = df.assign(risk_rank=[None for i in range(len(df))])

for une in unique_number_entered:
    df['risk_rank'].loc[df['number_entered'] ==
                        une] = unique_number_entered_rank[une]

df['risk_rank'] = pd.to_numeric(df['risk_rank'])

df[['number_entered', 'risk_rank']]

# # %%
# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('insurance_models_with_iv_loss_experience_2804.txt', 'w') as f:
#     f.write(summary_col([takeup_insurance_basic, takeup_insurance_controls,
#             wtp_basic, wtp_controls], stars=True, float_format='%0.4f').as_latex())
#     f.close()


# %
# create a ranking of participants based on how risk averse they are
df = df.assign(aversion_rank=[None for i in range(len(df))])
df['aversion_rank'] = df['number_entered'].rank(
    method='dense', ascending=False)

# %
df[['number_entered', 'aversion_rank']].sort_values(
    by=['number_entered'], ascending=False).values.tolist()
# %

# %
#######
# instead of number entered, use ranking of risk aversion
# as rank increases, risk aversion increases

# calculate wealth accumulated up until current point

# first calculate wealth in each period
df = df.assign(wealth_in_round=[None for i in range(len(df))])

for participant in participants:
    df['wealth_in_round'].loc[(df['label'] ==
                               participant) & (df['WTP'] >= df['insurance'])] = 100 - df['insurance'].loc[df['label'] == participant]
    df['wealth_in_round'].loc[(df['label'] ==
                               participant) & (df['WTP'] < df['insurance']) & (df['state'] == 'healthy')] = 100
    df['wealth_in_round'].loc[(df['label'] ==
                               participant) & (df['WTP'] < df['insurance']) & (df['state'] == 'ill')] = 0

df['wealth_in_round'] = pd.to_numeric(df['wealth_in_round'])
# %
# calculate wealth in previous round
df = df.assign(wealth_in_round_lag_1=[None for i in range(len(df))])

for participant in participants:
    df['wealth_in_round_lag_1'].loc[df['label'] ==
                                    participant] = df['wealth_in_round'].loc[df['label'] == participant].shift(1)

df['wealth_in_round_lag_1'] = pd.to_numeric(df['wealth_in_round_lag_1'])
# %
df = df.assign(cum_wealth=[None for i in range(len(df))])

# calculate accumulated wealth
for participant in participants:
    for round in [2, 3, 4, 5]:
        cum_wealth_li = []
        for i in range(1, round):
            cum_wealth_li.append(df['wealth_in_round'].loc[(df['label'] == participant) & (
                df['round_number'] == i)].values.tolist()[0])
        df['cum_wealth'].loc[(df['label'] == participant) & (
            df['round_number'] == round)] = sum(cum_wealth_li)

df['cum_wealth'] = pd.to_numeric(df['cum_wealth'])


# money leftover from investing in gneezy potters = 50 - number_entered
# result_risk is outcome/final payoff from gneezy potters
# result_risk = investedamount*return + 50 - number_entered
# investedamountreturn = result_risk - 50 + number_entered

df['gp_return'] = (df['result_risk']-50)/df['number_entered']
df['gp_return'] = df['gp_return'].fillna(0)
df['gp_return'] = df['gp_return'].map(lambda x: '%0.1f' % x)
df['gp_return'].unique()
df['gp_return'].value_counts()
df['gp_return'] = pd.to_numeric(df['gp_return'])
# %
number_entered_per_return = pd.DataFrame(df.groupby(['gp_return']).agg(
    {'number_entered': ['mean', 'var', 'count']})).reset_index()
# %
takeup_prob = probit(
    'insurance_takeup ~ treatment + prob + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number ', data=df).fit(cov_type='HC0')

takeup_prob_belief = probit(
    'insurance_takeup ~ treatment + prob +  belief  + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number ', data=df).fit(cov_type='HC0')

takeup_controls_prob = probit(
    'insurance_takeup ~ treatment + prob + belief  + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number + age + gender + smoker  + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

takeup_var = probit(
    'insurance_takeup ~ treatment+ var1 + var2 + var3 + var4 + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number ', data=df).fit(cov_type='HC0')

takeup_var_belief = probit(
    'insurance_takeup ~ treatment + var1 + var2 + var3 + var4+  belief  + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number ', data=df).fit(cov_type='HC0')

takeup_controls_var = probit(
    'insurance_takeup ~ treatment+ var1 + var2 + var3 + var4 + belief  + loss_amount + cum_wealth +  number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number + age + gender + smoker  + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('takeup_rate_110523.txt', 'w') as f:
#     f.write(summary_col([takeup_prob,takeup_prob_belief,takeup_controls_prob,takeup_var,takeup_var_belief,takeup_controls_var], stars=True, float_format='%0.4f').as_latex())
#     f.close()

# summary_col([takeup_prob,takeup_prob_belief,takeup_controls_prob,takeup_var,takeup_var_belief,takeup_controls_var], stars=True, float_format='%0.4f')
# %

wtp_prob = ols(
    'WTP ~ treatment + prob + loss_amount + cum_wealth   + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number', data=df).fit(cov_type='HC0')

wtp_prob_belief = ols(
    'WTP ~ treatment + prob + belief + loss_amount + cum_wealth    + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number', data=df).fit(cov_type='HC0')

# wtp with controls
wtp_controls_prob = ols(
    'WTP ~ treatment + prob + belief + loss_amount + cum_wealth  + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number + age + gender + smoker  + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

wtp_var = ols(
    'WTP ~ treatment + var1 + var2 + var3 + var4 + loss_amount + cum_wealth    + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number', data=df).fit(cov_type='HC0')

wtp_var_belief = ols(
    'WTP ~ treatment + var1 + var2 + var3 + var4 + belief + loss_amount + cum_wealth    + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number', data=df).fit(cov_type='HC0')

# wtp with controls
wtp_controls_var = ols(
    'WTP ~ treatment + var1 + var2 + var3 + var4 + belief + loss_amount + cum_wealth  + number_entered + gp_return + HL_switchpoint + timer_all_chars + round_number + age + gender + smoker  + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

# summary_col([wtp_prob,wtp_prob_belief,wtp_controls_prob,wtp_var,wtp_var_belief,wtp_controls_var], stars=True, float_format='%0.4f')

# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('wtp_110523.txt', 'w') as f:
#     f.write(summary_col([wtp_prob,wtp_prob_belief,wtp_controls_prob,wtp_var,wtp_var_belief,wtp_controls_var], stars=True, float_format='%0.4f').as_latex())
#     f.close()

# %
# check for multicollinearity
X_prob = add_constant(df[['prob', 'belief', 'loss_amount', 'cum_wealth', 'number_entered',
                      'result_risk', 'HL_switchpoint', 'timer_all_chars', 'round_number']]).dropna()
X_var = add_constant(df[['var1', 'var2', 'var3', 'var4', 'belief', 'loss_amount', 'cum_wealth',
                     'number_entered', 'result_risk', 'HL_switchpoint', 'timer_all_chars', 'round_number']]).dropna()

vif_prob = pd.Series([variance_inflation_factor(X_prob.values, i)
                     for i in range(X_prob.shape[1])], index=X_prob.columns)
vif_var = pd.Series([variance_inflation_factor(X_var.values, i)
                    for i in range(X_var.shape[1])], index=X_var.columns)

# %
# class IV2SLS(dependent, exog, endog, instruments, *, weights=None)[source]
# %
wtp_for_predicted_values_basic = ols(
    'WTP ~ belief + loss_amount + HL_switchpoint + number_entered + round_number + timer_all_chars', data=df).fit(cov_type='HC0')

wtp_for_predicted_values_basic.params

# %
df = df.assign(WTP_pred_basic=[None for i in range(len(df))])
df = df.assign(errors_basic=[None for i in range(len(df))])

for i in range(len(df)):
    df['WTP_pred_basic'].iloc[i] = np.dot(df[['belief', 'loss_amount', 'HL_switchpoint', 'number_entered', 'round_number',
                                          'timer_all_chars']].iloc[i], wtp_for_predicted_values_basic.params[1:]) + wtp_for_predicted_values_basic.params[0]
    df['errors_basic'].iloc[i] = df['WTP'].iloc[i] - np.dot(df[['belief', 'loss_amount', 'HL_switchpoint', 'number_entered', 'round_number',
                                                            'timer_all_chars']].iloc[i], wtp_for_predicted_values_basic.params[1:]) - wtp_for_predicted_values_basic.params[0]

df['WTP_pred_basic'] = pd.to_numeric(df['WTP_pred_basic'])
df['errors_basic'] = pd.to_numeric(df['errors_basic'])
# %
# # %
# first_stage = IV2SLS(df['belief'],
#                      df[['treatment', 'var1', 'var2', 'var3', 'var4']],
#                      None,
#                      None).fit(cov_type='robust')
# #first_stage.summary

# errors_instruments_basic = ols(
#     'errors_basic ~ C(treatment) + var1 + var2 + var3 + var4+ loss_amount + HL_switchpoint + number_entered+ number_entered + round_number + timer_all_chars', data=df).fit(cov_type='HC0')

# errors_instruments_basic.summary()

# treatment negative info is correlated with errors, therefore have to remove negative info
# as an instrument

# %
# create dummies for all treatments
treatments_excl_neginfo = ['baseline', 'fullinfo', 'posinfo', 'varinfo']

for treat in treatments_excl_neginfo:
    df[treat] = [0 for i in range(len(df))]

for treat in treatments_excl_neginfo:
    df[treat].mask(df['treatment'] == treat, 1, inplace=True)

# %
first_stage = IV2SLS(df['belief'],
                     df[['var1', 'var2', 'var3', 'var4']],
                     None,
                     None).fit(cov_type='robust')
# first_stage.summary

errors_instruments_basic = ols(
    'errors_basic ~  var1 + var2 + var3 + var4+ loss_amount + HL_switchpoint + number_entered+ number_entered + round_number + timer_all_chars', data=df).fit(cov_type='HC0')

# errors_instruments_basic.summary()


errors_instruments_controls = ols(
    'errors_basic ~  var1 + var2 + var3 + var4+ loss_amount + HL_switchpoint + number_entered+ number_entered + round_number + timer_all_chars + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

errors_instruments_controls.summary()
# cannot use TREATMENT as an instrument even if we consider only fullinfo,
# posinfo, baseline and varinfo, or if we only consider fullinfo, posinfo and varinfo
# %
# class IV2SLS(dependent, exog, endog, instruments, *, weights=None)[source]

wtp_iv_basic = IV2SLS(df['WTP'],
                      df[['number_entered', 'HL_switchpoint', 'loss_amount',
                          'cum_wealth', 'gp_return',
                          'round_number', 'timer_all_chars']],
                      df['belief'],
                      df[['var1', 'var2', 'var3', 'var4']]).fit(cov_type="robust")

# %
wtp_iv_controls = IV2SLS(df['WTP'],
                         df[['number_entered', 'HL_switchpoint', 'loss_amount', 'cum_wealth', 'round_number', 'gp_return', 'timer_all_chars',
                             'age', 'gender', 'smoker', 'health', 'exercisedays', 'exercisehours',
                             'insurancehealth', 'lifeinsurance', 'riskdriving', 'riskfinance',
                             'riskjob', 'riskhealth', 'risk']],
                         df['belief'],
                         df[['var1', 'var2', 'var3', 'var4']]).fit(cov_type="robust")
# %

wtp_iv_table = {'1': wtp_iv_basic,
                '2': wtp_iv_controls}

display(compare(wtp_iv_table))
print(compare(wtp_iv_table, precision='std_errors', stars=True).summary.as_latex())

# %
# WTP for different loss amounts
# takeup_basic_treatment = {}
# takeup_controls_treatment = {}
wtp_controls_treatment_prob = {}
wtp_controls_treatment_var = {}

for treat in treatment_names:
    # takeup_basic = probit('insurance_takeup ~ belief + number_entered + loss_amount + round_number + timer_all_chars + var1 + var2 + var3 + var4',
    #                       data=df.loc[df['treatment'] == treat]).fit()
    # takeup_basic_treatment[treat] = takeup_basic
    # takeup_controls = probit('insurance_takeup ~ belief  + loss_amount +  round_number + timer_all_chars + var1 + var2 + var3 + var4 + age + gender + smoker + aversion_rank + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df.loc[df['treatment']==treat]).fit()
    # takeup_controls_treatment[treat] = takeup_controls
    wtp_controls_prob = ols('WTP ~ prob + belief + loss_amount + number_entered + cum_wealth + gp_return + HL_switchpoint + round_number + timer_all_chars + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk',
                            data=df.loc[df['treatment'] == treat]).fit()
    wtp_controls_treatment_prob[treat] = wtp_controls_prob
    wtp_controls_var = ols('WTP ~ var1 + var2 + var3 + var4  + belief + loss_amount + number_entered + cum_wealth + gp_return + HL_switchpoint + round_number + timer_all_chars + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk',
                           data=df.loc[df['treatment'] == treat]).fit()
    wtp_controls_treatment_var[treat] = wtp_controls_var

# summary_col([(v) for k, v in takeup_basic_treatment.items()], stars=True)
# # %
os.chdir(base_directory+'/ols_s8_to_s17/')

wtp_treatment_for_latex = []
for i in range(len([(v) for k, v in wtp_controls_treatment_prob.items()])):
    wtp_treatment_for_latex.append(
        [(v) for k, v in wtp_controls_treatment_prob.items()][i])
    wtp_treatment_for_latex.append(
        [(v) for k, v in wtp_controls_treatment_var.items()][i])

# #%
# with open('wtp_by_treatment_120523.txt', 'w') as f:
#     f.write(summary_col(wtp_treatment_for_latex, stars=True, float_format='%0.4f').as_latex())
#     f.close()

# summary_col([(v) for k, v in wtp_controls_treatment_prob.items()], stars=True)

# summary_col([(v) for k, v in wtp_controls_treatment_var.items()], stars=True)

# %
# run IV for each treatment
wtp_basic_iv_treat = {}
wtp_controls_iv_treat = {}

wtp_iv_table_latex = {}

for treat in treatment_names:
    wtp_basic = IV2SLS(df['WTP'].loc[df['treatment'] == treat], df[['number_entered', 'HL_switchpoint', 'loss_amount', 'cum_wealth', 'gp_return', 'round_number', 'timer_all_chars']
                                                                   ].loc[df['treatment'] == treat], df['belief'].loc[df['treatment'] == treat], df[['var1', 'var2', 'var3', 'var4']].loc[df['treatment'] == treat]).fit()
    wtp_controls = IV2SLS(df['WTP'].loc[df['treatment'] == treat], df[['number_entered', 'HL_switchpoint', 'loss_amount', 'cum_wealth', 'round_number', 'gp_return', 'timer_all_chars', 'age', 'gender', 'smoker', 'health', 'exercisedays', 'exercisehours',
                          'insurancehealth', 'lifeinsurance', 'riskdriving', 'riskfinance', 'riskjob', 'riskhealth', 'risk']].loc[df['treatment'] == treat], df['belief'].loc[df['treatment'] == treat], df[['var1', 'var2', 'var3', 'var4']].loc[df['treatment'] == treat]).fit()
    wtp_basic_iv_treat[treat] = wtp_basic
    wtp_controls_iv_treat[treat] = wtp_controls
    wtp_iv_table_latex[treat + '_basic'] = wtp_basic
    wtp_iv_table_latex[treat + '_controls'] = wtp_controls

# %
display(compare(wtp_basic_iv_treat, precision='std_errors', stars=True))

# %
display(compare(wtp_controls_iv_treat, precision='std_errors', stars=True))

# %
print(compare(wtp_iv_table_latex, precision='std_errors',
      stars=True).summary.as_latex())


# %
# accuracy

accuracy_basic = ols(
    'abs_diff_belief_prob ~  treatment + loss_amount + number_entered + gp_return + cum_wealth + HL_switchpoint + round_number + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7', data=df).fit(cov_type='HC0')

accuracy_var = ols(
    'abs_diff_belief_prob ~ treatment +  var1 + var2 + var3 + var4 + loss_amount + number_entered + gp_return + cum_wealth + HL_switchpoint  + round_number + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7', data=df).fit(cov_type='HC0')

accuracy_controls_basic = ols(
    'abs_diff_belief_prob ~  treatment + loss_amount + number_entered + gp_return + cum_wealth + HL_switchpoint  + round_number + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

accuracy_controls_var = ols(
    'abs_diff_belief_prob ~  treatment + var1 + var2 + var3 + var4+ loss_amount + number_entered + gp_return + cum_wealth + HL_switchpoint  + round_number + timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7 + age + gender + smoker + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type='HC0')

# #%
# with open('accuracy_140523_i.txt', 'w') as f:
#     f.write(summary_col([accuracy_basic, accuracy_controls_basic, accuracy_var, accuracy_controls_var], stars=True, float_format='%0.4f').as_latex())
#     f.close()

summary_col([accuracy_basic, accuracy_controls_basic, accuracy_var,
            accuracy_controls_var], stars=True, float_format='%0.4f')
# %%

# %
model_5_prob = ols(
    'ln_prob ~ var1 + var2 + var3 + var4 - 1 ', data=df).fit(cov_type="HC0")

model_6_belief_basic = ols(
    'ln_belief ~ var1 + var2 + var3 + var4-1', data=df).fit(cov_type="HC0")


# %%

model_7_belief = ols(
    'ln_belief ~ treatment + var1 + var2 + var3 + var4 + round_number+ loss_amount + number_entered + timer_all_chars + cum_wealth + gp_return + HL_switchpoint ', data=df).fit(cov_type="HC0")

model_7_belief_interaction = ols(
    'ln_belief ~ treatment + var1 + var2 + var3 + var4 + round_number+ loss_amount + var1*timer_char1 + var2*timer_char2 + var3*timer_char3 + var4*timer_char4', data=df).fit(cov_type="HC0")

# model 7 plus participant dummies, dont use clustered errors

model_8_belief = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2 + var3 + var4 + round_number+ number_entered + loss_amount + timer_all_chars + cum_wealth + gp_return + HL_switchpoint', data=df).fit()

model_8_belief_interaction = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2 + var3 + var4 + round_number+ loss_amount + timer_all_chars + cum_wealth + gp_return + HL_switchpoint+ var1*timer_char1 + var2*timer_char2 + var3*timer_char3 + var4*timer_char4', data=df).fit()

# model 7 plus controls
# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint +
model_9_belief_controls = ols(
    'ln_belief ~ treatment + var1 + var2 + var3 + var4 + timer_all_chars + cum_wealth + gp_return + HL_switchpoint + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")

# model 8 plus controls
model_10_belief_controls = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2 + var3 + var4 + timer_all_chars + cum_wealth + gp_return + HL_switchpoint + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit()

# os.chdir(base_directory + '/ols_s8_to_s17/')
# with open('coef_recovery.txt', 'w') as f:
#     f.write(summary_col([model_5_prob, model_6_belief_basic, model_7_belief, model_8_belief, model_9_belief_controls, model_10_belief_controls], stars=True,float_format='%0.4f').as_latex())
#     f.close()

# %
model_6_belief_basic_var2_4_adj = ols(
    'ln_belief ~ var1 + var2_cat + var3 + var4_cat -1', data=df).fit(cov_type="HC0")

model_7_belief_var2_4_adj = ols(
    'ln_belief ~ treatment + var1 + var2_cat + var3 + var4_cat + round_number+ loss_amount + timer_all_chars + cum_wealth + gp_return + HL_switchpoint ', data=df).fit(cov_type="HC0")

# model 7 plus participant dummies, dont use clustered errors

model_8_belief_var2_4_adj = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2_cat + var3 + var4_cat + round_number+ loss_amount + timer_all_chars + cum_wealth + gp_return + HL_switchpoint', data=df).fit()

# model 7 plus controls
# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint +
model_9_belief_controls_var2_4_adj = ols(
    'ln_belief ~ treatment + var1 + var2_cat + var3 + var4_cat + timer_all_chars + cum_wealth + gp_return + HL_switchpoint + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit(cov_type="HC0")

# model 8 plus controls
model_10_belief_controls_var2_4_adj = ols(
    'ln_belief ~ treatment + C(label) + var1 + var2_cat + var3 + var4_cat + timer_all_chars + cum_wealth + gp_return + HL_switchpoint + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit()

ln_belief_table = summary_col([model_7_belief, model_9_belief_controls,
                              model_8_belief, model_10_belief_controls], stars=True, float_format='%0.4f')

# %%
print(ln_belief_table.tables[0][448:].to_latex())

# %%
##### plot participant betas ######

participant_beta_treat = pd.DataFrame(df.groupby(['treatment']).agg(
    {'coef_var1': ['mean', 'var'],
     'coef_var2': ['mean', 'var'],
     'coef_var3': ['mean', 'var'],
     'coef_var4': ['mean', 'var']}))


# %%
part_beta_dict = {}
for treat in treatment_names:
    part_beta_for_treat_dict = {}
    for coef in ['coef_var1', 'coef_var2', 'coef_var3', 'coef_var4']:
        part_beta_for_treat_dict[coef] = participant_beta_treat[coef]['mean'][treat]
    part_beta_dict[treat] = part_beta_for_treat_dict

part_beta_dict['true_beta'] = {
    'coef_var1': 0.7, 'coef_var2': 0.35, 'coef_var3': -0.6, 'coef_var4': -0.45}

part_beta_dict
# %%
part_beta_df = pd.DataFrame(part_beta_dict)
# %
colors_for_partbeta = ['darkgray', 'black', 'red', 'green', 'blue', 'black']
ind = np.arange(4)
width = 0.15
treatment_names_part_beta_for_display = ['Baseline',
                                         'Full info', 'Neg info', 'Pos info', 'All info', 'True Beta']
treatment_names_part_beta = ['baseline', 'fullinfo',
                             'neginfo', 'posinfo', 'varinfo', 'true_beta']

# %%
part_beta_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        part_beta_bars.append(plt.bar(
            ind+width*l, part_beta_df[treatment_names_part_beta[l]], width, color=colors_for_partbeta[l]))
    if l == 5:
        part_beta_bars.append(plt.bar(
            ind+width*l, part_beta_df[treatment_names_part_beta[l]], width, color=colors_for_partbeta[l], fill=False, hatch='///'))
    plt.xticks(ind+width+0.15, ['beta_1', 'beta_2', 'beta_3', 'beta_4'])

plt.axhline(y=0, xmin=-0.3, xmax=4.3, linestyle='-',
            lw=0.7, mfc='black', mec='black')
os.chdir(base_directory)
plt.xlabel('Characteristics')
plt.ylabel('Coefficient values')
# plt.legend(tuple(part_beta_bars), tuple(treatment_names_part_beta_for_display))
# plt.savefig('plots_s8_to_s17/part_betas_recovered.png')
plt.show()
# %%
# result of regressing on treatment subset
# regress log beliefs on var1 to 4 for each treatment
coef_names = ['beta_1', 'beta_2', 'beta_3', 'beta_4']

coefs_treat_subset_dict = {}
coefs_treat_subset_controls_dict = {}
coef_controls_params = {}
beliefs_subset = {}

for treat in treatment_names:
    coefs = ols('ln_belief ~ var1 + var2 + var3 + var4',
                data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0").params[1:].tolist()
    coef_controls = ols('ln_belief ~ var1 + var2 + var3 + var4 + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_all_chars + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk',
                        data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0").params.tolist()
    beliefs_subset[treat] = ols('ln_belief ~ var1 + var2 + var3 + var4 + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_all_chars + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0")
    treat_coefs = {}
    treat_coef_controls = {}
    for c in range(len(coefs)):
        treat_coefs[coef_names[c]] = coefs[c]
    for cc in range(1, 5):
        cc_min_one = cc-1
        treat_coef_controls[coef_names[cc_min_one]] = coef_controls[cc]
    coefs_treat_subset_dict[treat] = treat_coefs
    coefs_treat_subset_controls_dict[treat] = treat_coef_controls
    coef_controls_params[treat] = ols('ln_belief ~ treatment + var1 + var2 + var3 + var4 + round_number+ loss_amount + cum_wealth + gp_return + HL_switchpoint + timer_all_chars + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df.loc[df['treatment'] == treat]).fit(cov_type="HC0").params

coefs_treat_subset_dict['true_beta'] = {'beta_1': 0.7,
                                        'beta_2': 0.35,
                                        'beta_3': -0.6,
                                        'beta_4': -0.45}

coefs_treat_subset_controls_dict['true_beta'] = {'beta_1': 0.7,
                                                 'beta_2': 0.35,
                                                 'beta_3': -0.6,
                                                 'beta_4': -0.45}

coefs_treat_subset_df = pd.DataFrame(coefs_treat_subset_dict)
coefs_treat_subset_controls_df = pd.DataFrame(coefs_treat_subset_controls_dict)


os.chdir(base_directory + '/ols_s8_to_s17/')
with open('beliefs_subset.txt', 'w') as f:
    f.write(summary_col([beliefs_subset[treat] for treat in treatment_names],
            stars=True, float_format='%0.4f').as_latex())
    f.close()

# %%
coefs_treat_subset_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        coefs_treat_subset_bars.append(plt.bar(
            ind+width*l, coefs_treat_subset_df[treatment_names_part_beta[l]], width, color=colors_for_partbeta[l]))
    if l == 5:
        coefs_treat_subset_bars.append(plt.bar(
            ind+width*l, coefs_treat_subset_df[treatment_names_part_beta[l]], width, color=colors_for_partbeta[l], fill=False, hatch='///'))
    plt.xticks(ind+width+0.15, ['beta_1', 'beta_2', 'beta_3', 'beta_4'])

plt.axhline(y=0, xmin=-0.3, xmax=4.3, linestyle='-',
            lw=0.7, mfc='black', mec='black')
os.chdir(base_directory)
plt.xlabel('Characteristics')
plt.ylabel('Coefficient values')
plt.legend(tuple(part_beta_bars), tuple(treatment_names_part_beta_for_display))
# plt.savefig('plots_s8_to_s17/betas_subset_treat.png')
plt.show()

# %%
timer_dict = {}

for t in range(len(treatment_names)):
    timer_treat_dict = {}
    for timer in ['timer_char1', 'timer_char2', 'timer_char3', 'timer_char4', 'timer_char7', 'timer_all_chars']:
        timer_treat_dict[timer] = timer_sum_stats[timer]['mean'][t]
    timer_dict[treatment_names[t]] = timer_treat_dict

timer_df = pd.DataFrame(timer_dict)
timer_df

# %%

# %
ind = np.arange(6)
width = 0.15
timer_new_names = ['char1', 'char2', 'char3', 'char4', 'add info', 'all chars']
timer_bars = []

for l in range(len(treatment_names)):
    timer_bars.append(plt.bar(
        ind+width*l, timer_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, timer_new_names)

os.chdir(base_directory)
plt.axvline(x=4.775, linestyle='--', lw=0.7)
ax = plt.gca()
ax.set_ylim([0, 30])
plt.xlabel('Characteristics')
plt.ylabel('Time spent')
plt.legend(tuple(timer_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17\\timers.png')
plt.show()
# %%
diff_belief_prob_names = ['diff_belief_prob_x', 'diff_belief_prob_y']
diff_belief_prob_names_for_df = ['overestimated', 'underestimated']

under_over_mean = {}
under_over_var = {}
for t in range(len(treatment_names)):
    under_over_estimators_mean_dict = {}
    under_over_estimators_var_dict = {}
    for m in range(len(['diff_belief_prob_x', 'diff_belief_prob_y'])):
        under_over_estimators_mean_dict[diff_belief_prob_names_for_df[m]
                                        ] = moments_under_overestimators_df[diff_belief_prob_names[m]]['mean'][t]
        under_over_estimators_var_dict[diff_belief_prob_names_for_df[m]
                                       ] = moments_under_overestimators_df[diff_belief_prob_names[m]]['var'][t]
    under_over_mean[treatment_names[t]] = under_over_estimators_mean_dict
    under_over_var[treatment_names[t]] = under_over_estimators_var_dict

under_over_mean_df = pd.DataFrame(under_over_mean)
under_over_var_df = pd.DataFrame(under_over_var)


# %
ind = np.arange(2)
over_under_mean_bars = []

for l in range(len(treatment_names)):
    over_under_mean_bars.append(plt.bar(
        ind+width*l, under_over_mean_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, diff_belief_prob_names_for_df)

os.chdir(base_directory)
plt.axhline(y=0, linestyle='--', lw=0.7)
ax = plt.gca()
ax.set_ylim([-30, 30])
plt.ylabel('Difference between belief and probability, mean')
plt.legend(tuple(over_under_mean_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/diff_belief_prob_mean.png')
plt.show()
# %%
over_under_var_bars = []

for l in range(len(treatment_names)):
    over_under_var_bars.append(plt.bar(
        ind+width*l, under_over_var_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, diff_belief_prob_names_for_df)

os.chdir(base_directory)
plt.axhline(y=0, linestyle='--', lw=0.7)
ax = plt.gca()
ax.set_ylim([0, 500])
plt.ylabel('Difference between belief and probability, variance')
# plt.legend(tuple(over_under_var_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/diff_belief_prob_var.png')
plt.show()
# %%
df['true_exp_loss'] = df['prob']*df['loss_amount']
df['belief_exp_loss'] = df['belief_over_100']*df['loss_amount']

# %%
df[['belief_over_100', 'loss_amount', 'belief_exp_loss', 'WTP']]
# %%

# plot expected loss against WTP

# count = 0
# for treat in treatment_names:
#     plt.plot(df['true_exp_loss'].loc[(df['treatment'] == treat)], df['WTP'].loc[(df['treatment'] == treat)], markers[count], markersize=marker_size[count], mfc=colors[count], mec=colors[count], label=treatment_names_for_display[count])
#     plt.ylabel('Willingness to pay')
#     plt.xlabel('True expected loss')
#     plt.axline([0,0],[100,100], color='black')
#     # plt.set_xticks([])
#     count += 1
#     plt.show()


# os.chdir(base_directory)
# plt.show()
# %
# initialize subplot function
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
# loss_amounts = [20, 40, 60, 80, 100]
titles = ['Baseline', 'Full info', 'Neg info', 'Pos info', 'All info']

# %
for i in range(len(treatment_names)):
    loss = loss_amounts[i]
    axis[axes[i][0], axes[i][1]].plot(
        df['true_exp_loss'].loc[(df['treatment'] == treatment_names[i])], df['WTP'].loc[(df['treatment'] == treatment_names[i])], markers[i], markersize=marker_size[i], mfc=colors[i], mec=colors[i], label=treatment_names_for_display[i])
    axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
    axis[axes[i][0], axes[i][1]].axline(
        [0, 0], [100, 100], color='black', linestyle='dashed')
    axis[axes[i][0], axes[i][1]].xaxis.set_tick_params(labelbottom=False)
    axis[axes[i][0], axes[i][1]].set_xticks([])
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('WTP')
    if i == 4:
        axis[axes[i][0], axes[i][1]].set_xlabel('True expected loss')

os.chdir(base_directory)
figure.delaxes(axis[2, 1])
# plt.savefig('plots_s8_to_s17/WTP_true_exp_loss.png')
plt.show()
# %%

# initialize subplot function
figure, axis = plt.subplots(3, 2)
# axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
# # loss_amounts = [20, 40, 60, 80, 100]
# titles = ['Baseline', 'Full info', 'Neg info', 'Pos info', 'All info']

# %
for i in range(len(treatment_names)):
    loss = loss_amounts[i]
    axis[axes[i][0], axes[i][1]].plot(
        df['belief_exp_loss'].loc[(df['treatment'] == treatment_names[i])], df['WTP'].loc[(df['treatment'] == treatment_names[i])], markers[i], markersize=marker_size[i], mfc=colors[i], mec=colors[i], label=treatment_names_for_display[i])
    axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
    axis[axes[i][0], axes[i][1]].axline(
        [0, 0], [100, 100], color='black', linestyle='dashed')
    axis[axes[i][0], axes[i][1]].xaxis.set_tick_params(labelbottom=False)
    axis[axes[i][0], axes[i][1]].set_xticks([])
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('WTP')
    if i == 4:
        axis[axes[i][0], axes[i][1]].set_xlabel(
            'Expected loss, according to beliefs')

os.chdir(base_directory)
figure.delaxes(axis[2, 1])
# plt.savefig('plots_s8_to_s17/WTP_belief_exp_loss.png')
plt.show()

# %%

# how many WTP are above belief exp loss for each treatment

df = df.assign(wtp_above_belief_exp_loss=[0 for i in range(len(df))])

# %
df['wtp_above_belief_exp_loss'].mask(
    (df['WTP'] > df['belief_exp_loss']), 1, inplace=True)
df['wtp_above_belief_exp_loss'].mask(
    (df['WTP'] == df['belief_exp_loss']), 0, inplace=True)
df['wtp_above_belief_exp_loss'].mask(
    (df['WTP'] < df['belief_exp_loss']), -1, inplace=True)

df['wtp_above_belief_exp_loss'] = pd.to_numeric(
    df['wtp_above_belief_exp_loss'])
# %

# plot amount of wtp above belief couonts
wtp_above_belief_exp_loss_df = df.groupby(
    ['wtp_above_belief_exp_loss', 'treatment']).agg(
        {'WTP': ['mean', 'count']}
)

wtp_above_belief_exp_loss_df

# %
wtp_above_belief_exp_loss_dict = {}  # count
wtp_belief_exp_loss_dict = {}  # mean
for treat in treatment_names:
    wtp_above_belief_exp_loss_treat_dict = {}
    wtp_belief_exp_loss_treat_dict = {}
    for dummy in [-1, 0, 1]:
        wtp_above_belief_exp_loss_treat_dict[str(
            dummy)] = wtp_above_belief_exp_loss_df['WTP']['count'][dummy][treat]
        wtp_belief_exp_loss_treat_dict[str(
            dummy)] = wtp_above_belief_exp_loss_df['WTP']['mean'][dummy][treat]
    wtp_above_belief_exp_loss_dict[treat] = wtp_above_belief_exp_loss_treat_dict
    wtp_belief_exp_loss_dict[treat] = wtp_belief_exp_loss_treat_dict

# %
wtp_above_exp_belief = pd.DataFrame(wtp_above_belief_exp_loss_dict)
wtp_belief_exp_loss = pd.DataFrame(wtp_belief_exp_loss_dict)

# %
colors_for_wtpaboveexpbelief = ['darkgray',
                                'black', 'red', 'green', 'blue', 'black']
ind = np.arange(3)
width = 0.15
treatment_names_for_display = ['Baseline',
                               'Full info', 'Neg info', 'Pos info', 'All info']
treatment_names_display = ['baseline', 'fullinfo',
                           'neginfo', 'posinfo', 'varinfo']

wtpaboveexpbelief_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        wtpaboveexpbelief_bars.append(plt.bar(
            ind+width*l, wtp_above_exp_belief[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l]))
    # if l == 5:
    #     wtpaboveexpbelief_bars.append(plt.bar(
    #         ind+width*l, wtp_above_exp_belief[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l], fill=False, hatch='///'))
    plt.xticks(ind+width+0.15, ['WTP < exp loss',
               'WTP = exp loss', 'WTP > exp loss'])

# plt.axhline(y=0, xmin=-0.3, xmax=4.3, linestyle='-',
#             lw=0.7, mfc='black', mec='black')
os.chdir(base_directory)
plt.ylabel('Count, WTP vs expected loss based on beliefs')
plt.legend(tuple(wtpaboveexpbelief_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_above_exp_loss_belief.png')
plt.show()

# %

wtpexpbelief_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        wtpexpbelief_bars.append(plt.bar(
            ind+width*l, wtp_belief_exp_loss[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l]))
    # if l == 5:
    #     wtpaboveexpbelief_bars.append(plt.bar(
    #         ind+width*l, wtp_above_exp_belief[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l], fill=False, hatch='///'))
    plt.xticks(ind+width+0.15, ['WTP < exp loss',
               'WTP = exp loss', 'WTP > exp loss'])

# plt.axhline(y=0, xmin=-0.3, xmax=4.3, linestyle='-',
#             lw=0.7, mfc='black', mec='black')
os.chdir(base_directory)
plt.ylabel('Mean, WTP vs expected loss based on beliefs')
plt.legend(tuple(wtpexpbelief_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_above_exp_loss_belief_mean.png')
plt.show()

# %%

# mann whitney U to see whether count of wtp above differs from below or equal
mannwhitneyu_wtp_above_exp_loss_belief = {}
for treat in treatment_names:
    mannwhitneyu_wtp_above_exp_loss_belief_dummy = {}
    for dummy in [-1, 0, 1]:
        other_dummies = []
        dummy_stat_p = {}
        for od in [-1, 0, 1]:
            if od != dummy:
                other_dummies.append(od)
        for other_dummy in other_dummies:
            stat_p, p_t = mannwhitneyu(df['WTP'].loc[(df['wtp_above_belief_exp_loss'] == dummy) & (
                df['treatment'] == treat)], df['WTP'].loc[(df['wtp_above_belief_exp_loss'] == other_dummy) & (df['treatment'] == treat)])
            dummy_stat_p[other_dummy] = [stat_p, p_t]
        mannwhitneyu_wtp_above_exp_loss_belief_dummy[dummy] = dummy_stat_p
    mannwhitneyu_wtp_above_exp_loss_belief[treat] = mannwhitneyu_wtp_above_exp_loss_belief_dummy

mannwhitneyu_wtp_above_exp_loss_belief
# %%

#### plot abs diff belief and prob ###

abs_diff_belief_prob_dict = {}

for t in range(len(treatment_names)):
    abs_diff_belief_prob_dict[treatment_names[t]
                              ] = var_by_treatment_df['abs_diff_belief_prob']['mean'][t]

abs_diff_belief_prob_df = pd.DataFrame(abs_diff_belief_prob_dict, index=[0])
abs_diff_belief_prob_df

# %%
sum_abs_diff_belief_prob = 0
for k, v in abs_diff_belief_prob_dict.items():
    sum_abs_diff_belief_prob += v

avg_abs_diff_belief_prob = sum_abs_diff_belief_prob/5
avg_abs_diff_belief_prob
# %%
# %
ind = np.arange(1)
width = 0.1
abs_diff_belief_prob_bars = []

for l in range(len(treatment_names)):
    abs_diff_belief_prob_bars.append(plt.bar(
        ind+width*l, abs_diff_belief_prob_df[treatment_names[l]], width, color=colors_for_wtp[l], label=treatment_names_for_display[l]))
    # plt.xticks(ind+width+0.15, treatment_names_for_display)

os.chdir(base_directory)
plt.axhline(y=avg_abs_diff_belief_prob, linestyle='--', lw=1.0, color='black')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
ax.set_ylim([0, 40])
# plt.xlabel('Characteristics')
plt.ylabel('Absolaute difference between belief and probability')
plt.legend(tuple(abs_diff_belief_prob_bars), tuple(
    treatment_names_for_display), loc='lower right')
# plt.savefig('plots_s8_to_s17/mean_abs_diff_belief_prob.png')
plt.show()
# %
### count of those who overestimated and underestimated ###

diff_belief_prob_names = ['diff_belief_prob_x', 'diff_belief_prob_y']
diff_belief_prob_names_for_df = ['overestimated', 'underestimated']

over_under_count = {}
for t in range(len(treatment_names)):
    over_under_count_dict = {}
    for m in range(len(['diff_belief_prob_x', 'diff_belief_prob_y'])):
        over_under_count_dict[diff_belief_prob_names_for_df[m]
                              ] = moments_under_overestimators_df[diff_belief_prob_names[m]]['count'][t]
    over_under_count[treatment_names[t]] = over_under_count_dict

over_under_count_df = pd.DataFrame(over_under_count)


# %
ind = np.arange(2)
over_under_count_bars = []

for l in range(len(treatment_names)):
    over_under_count_bars.append(plt.bar(
        ind+width*l, over_under_count_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, diff_belief_prob_names_for_df)

os.chdir(base_directory)
plt.axhline(y=0, linestyle='--', lw=0.7)
ax = plt.gca()
# ax.set_ylim([-30, 30])
plt.ylabel('Count, abs diff belief and prob')
plt.legend(tuple(over_under_count_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/overestimator_counts.png')
plt.show()
# %%

# mann whitney u test to see if there is a difference in accuracy between
# the over and underestimators in each treatment

df = df.assign(overestimator=[0 for i in range(len(df))])
df['overestimator'].mask((df['diff_belief_prob'] > 0), 1, inplace=True)
df['overestimator'] = pd.to_numeric(df['overestimator'])
# %%
mannwhitneyu_overestimator = {}
for treat in treatment_names:
    mannwhitneyu_overestimator_dummy = {}
    for dummy in [0, 1]:
        other_dummies = []
        dummy_stat_p = {}
        for od in [0, 1]:
            if od != dummy:
                other_dummies.append(od)
        for other_dummy in other_dummies:
            stat_p, p_t = mannwhitneyu(df['abs_diff_belief_prob'].loc[(df['overestimator'] == dummy) & (
                df['treatment'] == treat)], df['abs_diff_belief_prob'].loc[(df['overestimator'] == other_dummy) & (df['treatment'] == treat)])
            dummy_stat_p[other_dummy] = [stat_p, p_t]
        mannwhitneyu_overestimator_dummy[dummy] = dummy_stat_p
    mannwhitneyu_overestimator[treat] = mannwhitneyu_overestimator_dummy

mannwhitneyu_overestimator
# %%


# wtp for over and underestimators

wtp_split = pd.DataFrame(df.groupby(['treatment', 'overestimator']).agg(
    {'WTP': ['mean', 'var']}))

overestimated_names = ['overestimated', 'underestimated']

wtp_split_dict = {}
for treat in treatment_names:
    wtp_split_treat_dict = {}
    for underestimated in [0, 1]:
        overestimated = 1-underestimated
        wtp_split_treat_dict[overestimated_names[underestimated]
                             ] = wtp_split['WTP']['mean'][treat][overestimated]
    wtp_split_dict[treat] = wtp_split_treat_dict

wtp_split_df = pd.DataFrame(wtp_split_dict)

wtp_split_df

# %%


ind = np.arange(2)
wtp_split_bars = []

for l in range(len(treatment_names)):
    wtp_split_bars.append(plt.bar(
        ind+width*l, wtp_split_df[treatment_names[l]], width, color=colors_for_wtp[l]))
    plt.xticks(ind+width+0.15, overestimated_names)

os.chdir(base_directory)
plt.axhline(y=0, linestyle='--', lw=0.7)
ax = plt.gca()
# ax.set_ylim([-30, 30])
plt.ylabel('Willingness to pay')
plt.legend(tuple(wtp_split_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/overestimator_WTP.png')
plt.show()


# %%

# mannwhitneyu test of difference in WTP between over and underestimators

mannwhitneyu_wtp_overestimator = {}
for treat in treatment_names:
    mannwhitneyu_overestimator_dummy = {}
    for dummy in [0, 1]:
        other_dummies = []
        dummy_stat_p = {}
        for od in [0, 1]:
            if od != dummy:
                other_dummies.append(od)
        for other_dummy in other_dummies:
            stat_p, p_t = mannwhitneyu(df['WTP'].loc[(df['overestimator'] == dummy) & (
                df['treatment'] == treat)], df['WTP'].loc[(df['overestimator'] == other_dummy) & (df['treatment'] == treat)])
            dummy_stat_p[other_dummy] = [stat_p, p_t]
        mannwhitneyu_overestimator_dummy[dummy] = dummy_stat_p
    mannwhitneyu_wtp_overestimator[treat] = mannwhitneyu_overestimator_dummy

mannwhitneyu_wtp_overestimator
# %%


##################

# %%

# how many WTP are above true prob exp loss for each treatment

df = df.assign(wtp_above_prob_exp_loss=[0 for i in range(len(df))])

# %
df['wtp_above_prob_exp_loss'].mask(
    (df['WTP'] > df['true_exp_loss']), 1, inplace=True)
df['wtp_above_prob_exp_loss'].mask(
    (df['WTP'] == df['true_exp_loss']), 0, inplace=True)
df['wtp_above_prob_exp_loss'].mask(
    (df['WTP'] < df['true_exp_loss']), -1, inplace=True)

df['wtp_above_prob_exp_loss'] = pd.to_numeric(
    df['wtp_above_prob_exp_loss'])
# %

# plot amount of wtp above true prob exp loss counts and mean
wtp_above_prob_exp_loss_df = df.groupby(
    ['wtp_above_prob_exp_loss', 'treatment']).agg(
        {'WTP': ['mean', 'count']}
)

wtp_above_prob_exp_loss_df

# %
wtp_above_prob_exp_loss_dict = {}  # count
wtp_prob_exp_loss_dict = {}  # mean
for treat in treatment_names:
    wtp_above_prob_exp_loss_treat_dict = {}
    wtp_prob_exp_loss_treat_dict = {}
    for dummy in [-1, 1]:
        wtp_above_prob_exp_loss_treat_dict[str(
            dummy)] = wtp_above_prob_exp_loss_df['WTP']['count'][dummy][treat]
        wtp_prob_exp_loss_treat_dict[str(
            dummy)] = wtp_above_prob_exp_loss_df['WTP']['mean'][dummy][treat]
    wtp_above_prob_exp_loss_dict[treat] = wtp_above_prob_exp_loss_treat_dict
    wtp_prob_exp_loss_dict[treat] = wtp_prob_exp_loss_treat_dict

# %
wtp_above_exp_prob = pd.DataFrame(wtp_above_prob_exp_loss_dict)
wtp_prob_exp_loss = pd.DataFrame(wtp_prob_exp_loss_dict)

# %
colors_for_wtpaboveexpbelief = ['darkgray',
                                'black', 'red', 'green', 'blue', 'black']
ind = np.arange(2)
width = 0.15
treatment_names_for_display = ['Baseline',
                               'Full info', 'Neg info', 'Pos info', 'All info']
treatment_names_display = ['baseline', 'fullinfo',
                           'neginfo', 'posinfo', 'varinfo']

wtpaboveexpprob_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        wtpaboveexpprob_bars.append(plt.bar(
            ind+width*l, wtp_above_exp_prob[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l]))
    plt.xticks(ind+width+0.15, ['WTP < exp loss', 'WTP > exp loss'])

os.chdir(base_directory)
plt.ylabel('Count, WTP vs expected loss based on probability')
plt.legend(tuple(wtpaboveexpprob_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_above_exp_loss_prob.png')
plt.show()

# %%

wtpexpprob_bars = []
for l in range(len(treatment_names_part_beta)):
    if l != 5:
        wtpexpprob_bars.append(plt.bar(
            ind+width*l, wtp_prob_exp_loss[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l]))
    plt.xticks(ind+width+0.15, ['WTP < exp loss', 'WTP > exp loss'])

# plt.axhline(y=0, xmin=-0.3, xmax=4.3, linestyle='-',
#             lw=0.7, mfc='black', mec='black')
os.chdir(base_directory)
plt.ylim(0, 50)
plt.ylabel('Mean, WTP vs expected loss based on probability')
plt.legend(tuple(wtpexpbelief_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_above_exp_loss_prob_mean.png')
plt.show()
# %%


# mannwhitney u test for difference in accuracy between treatments


mannwhitneyu_accuracy_results = {}
for treat in treatment_names:
    other_treatments = []
    treat_stat_p = {}
    mannwhitneyu_accuracy_result_treat = {}
    for ot in treatment_names:
        if ot != treat:
            other_treatments.append(ot)
    for other_treat in other_treatments:
        stat_t, p_t = mannwhitneyu(df['abs_diff_belief_prob'].loc[(
            df['treatment'] == treat)], df['abs_diff_belief_prob'].loc[(df['treatment'] == other_treat)])
        treat_stat_p[other_treat] = [stat_t, p_t]
    mannwhitneyu_accuracy_results[treat] = treat_stat_p

mannwhitneyu_accuracy_results
# %%
#############################################

# wtp vs expected loss by loss amount #

#############################################

# plot amount of wtp above true prob exp loss counts and mean
wtp_belief_exp_loss_by_loss_df = df.groupby(
    ['loss_amount', 'wtp_above_belief_exp_loss', 'treatment']).agg(
        {'WTP': ['mean', 'count']}
).unstack(fill_value=0).stack()

wtp_belief_exp_loss_by_loss_df

mean_wtp_vs_exploss_by_loss_dict = {}
count_wtp_vs_exploss_by_loss_dict = {}
for loss in loss_amounts:
    loss_mean_dict = {}
    loss_count_dict = {}
    for treat in treatment_names:
        treat_mean_dict = {}
        treat_count_dict = {}
        for dummy in [-1, 0, 1]:
            treat_mean_dict[str(
                dummy)] = wtp_belief_exp_loss_by_loss_df['WTP']['mean'][loss][dummy][treat]
            treat_count_dict[str(
                dummy)] = wtp_belief_exp_loss_by_loss_df['WTP']['count'][loss][dummy][treat]
        loss_mean_dict[treat] = treat_mean_dict
        loss_count_dict[treat] = treat_count_dict
    mean_wtp_vs_exploss_by_loss_dict[loss] = pd.DataFrame(loss_mean_dict)
    count_wtp_vs_exploss_by_loss_dict[loss] = pd.DataFrame(loss_count_dict)

# %%
colors_for_wtpaboveexpbelief = ['darkgray',
                                'black', 'red', 'green', 'blue', 'black']
ind = np.arange(2)
width = 0.15
treatment_names_for_display = ['Baseline',
                               'Full info', 'Neg info', 'Pos info', 'All info']
treatment_names_display = ['baseline', 'fullinfo',
                           'neginfo', 'posinfo', 'varinfo']

wtpvsexpbelief_bars = []
for l in range(len(treatment_names)):
    if l != 5:
        wtpaboveexpprob_bars.append(plt.bar(
            ind+width*l, wtp_above_exp_prob[treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l]))
    plt.xticks(ind+width+0.15, ['WTP < exp loss','WTP > exp loss'])

os.chdir(base_directory)
plt.ylabel('Count, WTP vs expected loss based on probability')
plt.legend(tuple(wtpaboveexpprob_bars), tuple(treatment_names_for_display))
# plt.savefig('plots_s8_to_s17/wtp_above_exp_loss_prob.png')
plt.show()
# %%
ind = np.arange(3)
width = 0.15
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
loss_amounts = [20, 40, 60, 80, 100]
titles = ['Loss = 20', 'Loss = 40', 'Loss = 60', 'Loss = 80', 'Loss = 100']

wtpvsexpbelief_loss_bars = []
for i in range(len(loss_amounts)):
    loss = loss_amounts[i]
    for l in range(len(treatment_names)):
        if l != 5:
            axis[axes[i][0], axes[i][1]].bar(ind+width*l, mean_wtp_vs_exploss_by_loss_dict[loss][treatment_names_display[l]],
                                             width, color=colors_for_wtpaboveexpbelief[l], label=treatment_names_for_display[l])
            axis[axes[i][0], axes[i][1]].set_ylim([0,80])
        # axis[axes[i][0], axes[i][1]].plot(df['prob'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
        axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
        plt.xticks(ind+width+0.15, ['WTP < exp loss',
                   'WTP = exp loss', 'WTP > exp loss'])
        # axis[axes[i][0], axes[i][1]].axhline(0.5*loss, linestyle='--', lw=0.5)
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('Willingness to pay')
        wtpvsexpbelief_loss_bars.append(axis[axes[i][0], axes[i][1]].bar(ind+width*l, mean_wtp_vs_exploss_by_loss_dict[loss]
                                        [treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l], label=treatment_names_for_display[l]))
    if i == 4:
        # axis[axes[i][0], axes[i][1]].set_xlabel('True probability')
        axis[axes[i][0], axes[i][1]].legend(bbox_to_anchor=(2, 1))
    if (i != 4) & (i != 3):
        axis[axes[i][0], axes[i][1]].set_xticks([])
    if (i == 4) & (i == 3):
        # axis[axes[i][0], axes[i][1]].xaxis.set_xticks([ind+width+0.15], labels=['WTP < exp loss', 'WTP = exp loss', 'WTP > exp loss'])
        # plt.xticks(ind+width+0.15, ['WTP < exp loss', 'WTP = exp loss', 'WTP > exp loss'])
        plt.legend(tuple(wtpvsexpbelief_loss_bars),
                   tuple(treatment_names_for_display))

plt.xticks(ind+width+0.15, ['WTP < exp loss',
           'WTP = exp loss', 'WTP > exp loss'])
figure.delaxes(axis[2, 1])
os.chdir(base_directory)
# plt.savefig('plots_s8_to_s17/wtp_per_prob_loss_treat.png')
plt.show()

# %%

df = df.assign(belief_cat=[None for i in range(len(df))])
for belief in [20,40,60,80,100]:
    start_point = belief - 20
    df['belief_cat'].mask((df['belief'] > start_point) & (df['belief']<=belief), belief, inplace=True)

df[['belief','belief_cat']]
# %%

wtp_belief_exp_loss_by_belief_df = df.groupby(
    ['belief_cat', 'wtp_above_belief_exp_loss', 'treatment']).agg(
        {'WTP': ['mean', 'count']}
).unstack(fill_value=0).stack()


mean_wtp_vs_exploss_by_belief_dict = {}
count_wtp_vs_exploss_by_belief_dict = {}
for cat in [20,40,60,80,100]:
    cat_mean_dict = {}
    cat_count_dict = {}
    for treat in treatment_names:
        treat_mean_dict = {}
        treat_count_dict = {}
        for dummy in [-1, 0, 1]:
            treat_mean_dict[str(
                dummy)] = wtp_belief_exp_loss_by_belief_df['WTP']['mean'][cat][dummy][treat]
            treat_count_dict[str(
                dummy)] = wtp_belief_exp_loss_by_belief_df['WTP']['count'][cat][dummy][treat]
        loss_mean_dict[treat] = treat_mean_dict
        loss_count_dict[treat] = treat_count_dict
    mean_wtp_vs_exploss_by_belief_dict[cat] = pd.DataFrame(loss_mean_dict)
    count_wtp_vs_exploss_by_belief_dict[cat] = pd.DataFrame(loss_count_dict)



#%%
# plot wtp above and below exp loss but for different belief categories
ind = np.arange(3)
width = 0.15
figure, axis = plt.subplots(3, 2)
axes = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
belief_cats = [20,40,60,80,100]
titles = ['Beliefs between 0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%']

plt.setp(axis, xticks=[0.275,1.275,2.275], xticklabels=['WTP < exp loss','WTP = exp loss', 'WTP > exp loss'])

for ax in axis.flatten():
    plt.sca(ax)
    plt.xticks(rotation=45)

wtpvsexpbelief_cat_bars = []
for i in range(len(belief_cats)):
    cat = belief_cats[i]
    for l in range(len(treatment_names)):
        if l != 5:
            axis[axes[i][0], axes[i][1]].bar(ind+width*l, mean_wtp_vs_exploss_by_belief_dict[cat][treatment_names_display[l]],
                                             width, color=colors_for_wtpaboveexpbelief[l], label=treatment_names_for_display[l])
            axis[axes[i][0], axes[i][1]].set_ylim([0,80])
        # axis[axes[i][0], axes[i][1]].plot(df['prob'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], df['WTP'].loc[(df['treatment'] == treatment_names[l]) & (df['loss_amount'] == loss)], markers[l], markersize=marker_size[l], mfc=colors[l], mec=colors[l], label=treatment_names_for_display[l])
        axis[axes[i][0], axes[i][1]].set_title(titles[i], fontsize=10)
        axis[axes[i][0], axes[i][1]].set_yticks([0,20,40,60,80])
        # axis[axes[i][0], axes[i][1]].axhline(0.5*loss, linestyle='--', lw=0.5)
    if i == 0:
        axis[axes[i][0], axes[i][1]].set_ylabel('Willingness to pay')
        wtpvsexpbelief_cat_bars.append(axis[axes[i][0], axes[i][1]].bar(ind+width*l, mean_wtp_vs_exploss_by_loss_dict[loss]
                                        [treatment_names_display[l]], width, color=colors_for_wtpaboveexpbelief[l], label=treatment_names_for_display[l]))
    if i == 4:
        # axis[axes[i][0], axes[i][1]].set_xlabel('True probability')
        axis[axes[i][0], axes[i][1]].legend(bbox_to_anchor=(2, 1))
    if (i != 4) & (i != 3):
        axis[axes[i][0], axes[i][1]].set_xticks([])
    if (i == 4) & (i == 3):
        # axis[axes[i][0], axes[i][1]].xaxis.set_xticks([ind+width+0.15], labels=['WTP < exp loss', 'WTP = exp loss', 'WTP > exp loss'])
        # plt.xticks(ind+width+0.15, ['WTP < exp loss', 'WTP = exp loss', 'WTP > exp loss'])
        plt.legend(tuple(wtpvsexpbelief_cat_bars),
                   tuple(treatment_names_for_display))

figure.delaxes(axis[2, 1])
os.chdir(base_directory)
# plt.savefig('plots_s8_to_s17/wtp_per_prob_loss_treat.png')
plt.show()
# %%
