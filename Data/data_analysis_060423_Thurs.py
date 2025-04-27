# %%
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
# base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'
base_directory = 'U:\\documents\\Chapter_2\\Data\\raw\\'
os.chdir(base_directory)

data = []
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

df['timer_all_chars'] = (df['timer_char1'] + df['timer_char2'] +
                         df['timer_char3'] + df['timer_char4'])/1000

df = df.reset_index()


# find variance for each probability decile
df['decile'] = pd.qcut(df['prob'], q=10, labels=False)
# %
data_belief_prob_treatment = pd.DataFrame(df.groupby(['decile']).agg({'diff_belief_prob': ['mean', 'var'],
                                                                      'belief': ['mean', 'var'],
                                                                      'prob': ['mean', 'var']})).reset_index()

#
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

# %

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
                        'var_prob', 'neg_prob', 'pos_prob',  'treatment']


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
# plt.savefig('filtered_plots_s8_to_s17/accuracy_abs_diff_treatment_histogram.png')
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
# center variables to avoid large condition number error
# df['prob'] -= np.average(df['prob'])
# # df['diff_belief_prob'] -= np.average(df['diff_belief_prob'])
# # df['belief'] -= np.average(df['belief'])
# df['abs_diff_belief_prob'] -= np.average(df['abs_diff_belief_prob'])

# %

# basic accuracy model
model_0_accuracy_filtered = ols(
    'abs_diff_belief_prob ~   C(treatment) + prob + round_number + loss_amount', data=df).fit()  # fit(cov_type="HC1")
# %
model_0_accuracy_filtered.summary()

# %%

# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint
# basic accuracy model + controls
model_0_accuracy_filtered_controls = ols(
    'abs_diff_belief_prob ~   C(treatment) + prob  + round_number + loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk', data=df).fit()  # .fit()#fit(cov_type="HC0")
# %
model_0_accuracy_filtered_controls.summary()

# %%
# basic accuracy model with prob and prob^2
model_1_accuracy_filtered = ols(
    'abs_diff_belief_prob ~ C(treatment) + prob  + round_number + loss_amount+' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")
# %
model_1_accuracy_filtered.summary()

# %%
# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint
# basic accuracy model with prob and prob^2 and controls
model_1_accuracy_filtered_controls = ols(
    'abs_diff_belief_prob ~ C(treatment) + prob  + round_number + loss_amount + gender + smoker + number_entered + + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + ' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")
# %
model_1_accuracy_filtered_controls.summary()
# %%

# basic accuracy model with vars
model_2_accuracy_filtered = ols(
    'abs_diff_belief_prob  ~ C(treatment)  + var1 +var2 +var3 + var4 + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")
# %
model_2_accuracy_filtered.summary()

# %

# basic accuracy model with vars and controls
model_2_accuracy_filtered_controls = ols(
    'abs_diff_belief_prob  ~ C(treatment)  + var1 +var2 +var3 + var4 + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint', data=df).fit()  # fit(cov_type="HC0")
# %
model_2_accuracy_filtered_controls.summary()

# %
# basic accuracy model with prob and prob^2
model_3_accuracy_filtered = ols(
    'abs_diff_belief_prob ~ C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount +' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")
# %
model_3_accuracy_filtered.summary()

# cannot do the equivalent of this model with prob^2 due to multicollinearity
# %

# check how beliefs are biased in the basic model
# ie see if the diff is pos or neg for each treatment
model_4_diff = ols(
    'diff_belief_prob ~ C(treatment) + prob  + round_number+ loss_amount + ' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_4_diff.summary()

# no effect
# %
# same as model 4 but with var

model_5_diff = ols(
    'diff_belief_prob ~ C(treatment) + prob  + var1 + var2 + var3 + var4 + round_number+ loss_amount + ' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_5_diff.summary()


# %%
# %
# create ln belief/(1-belief)

df['belief_over_100'] = df['belief']/100
df['ln_belief'] = np.log(
    df['belief_over_100']/(1-df['belief_over_100']))
df['ln_prob'] = np.log(df['prob']/(1-df['prob']))

# %
# get rid of observations with value of ln_belief == inf/-inf
# ie get rid of observations where belief_over_100 = 0 or 1
df.loc[df['ln_belief'] == float('inf'), 'ln_belief'] = None
df.loc[df['ln_belief'] == -float('inf'), 'ln_belief'] = None
# %%
# %
model_6_prob = ols(
    'ln_prob ~ C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_6_prob.summary()


# %%
model_7_belief = ols(
    'ln_belief ~ C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount +' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_7_belief.summary()

# %%
# model 7 plus controls
# + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint +
model_7_belief_controls = ols(
    'ln_belief ~ C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint + ' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_7_belief_controls.summary()
# %
# regress on dummies for each participant
model_8_belief = ols(
    'ln_belief ~ C(label) + C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount + ' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_8_belief.summary()

# %
# regress on dummies for each participant + controls
model_8_belief_controls = ols(
    'ln_belief ~ C(label) + C(treatment)  + var1 + var2 + var3 + var4 + round_number+ loss_amount + age + gender + smoker + number_entered + health + exercisedays + exercisehours + insurancehealth + lifeinsurance + riskdriving + riskfinance + riskjob + riskhealth + risk + HL_switchpoint + HLLF_switchpoint +' + 'I(prob**2)', data=df).fit()  # fit(cov_type="HC0")

model_8_belief_controls.summary()

# %

os.chdir(base_directory+'ols_s8_to_s17/')

models_filtered_names = [
    'model_0_accuracy_filtered',
    'model_1_accuracy_filtered',
    'model_2_accuracy_filtered',
    'model_3_accuracy_filtered',
    'model_4_diff',
    'model_5_diff',
    'model_6_prob',
    'model_7_belief',
    'model_8_belief'
]

models_filtered = [
    model_0_accuracy_filtered,
    model_1_accuracy_filtered,
    model_2_accuracy_filtered,
    model_3_accuracy_filtered,
    model_4_diff,
    model_5_diff,
    model_6_prob,
    model_7_belief,
    model_8_belief
]

for m in range(len(models_filtered)):
    with open(models_filtered_names[m] + '.txt', 'w') as f:
        for table in models_filtered[m].summary().tables:
            f.write(
                table.as_latex_tabular()
            )
    f.close()


# %
model_belief_ols = IV2SLS(df['ln_belief'], df[[
    'var1', 'var2', 'var3', 'var4', 'number_entered', 'treatment']], None, None).fit(cov_type='robust')

model_belief_2sls = IV2SLS(df['ln_belief'], df[['var1', 'var2', 'var3', 'var4', 'treatment']],
                           df['number_entered'], df[['gender', 'smoker', 'exercisedays', 'exercisehours']]).fit(cov_type="robust")

print(model_belief_2sls)

print(compare({'OLS': model_belief_ols, '2SLS': model_belief_2sls}))

# %%
df[['coef_var1', 'coef_var2', 'coef_var3', 'coef_var4',
    'beta_1', 'beta_2', 'beta_3', 'beta_4']] = [0, 0, 0, 0, 0, 0, 0, 0]

# %
# construct difference between participant coefficients and true coefficients
df[['diff_var1', 'diff_var2',
    'diff_var3', 'diff_var4']] = [0, 0, 0, 0]

# %%
# regress on each participant

# keep only participants which appear at least twice
# this is needed for regression as you cant regress with just one observation
p_counts = df['label'].value_counts()
df = df[df.label.isin(
    p_counts.index[p_counts.gt(1)])]
participant_labels = df['label'].unique().tolist()
# %
for participant in participant_labels:
    participant_data = df.loc[df['label'] == participant]
    model_belief_participant = ols(
        'ln_belief ~ var1 + var2 + var3 + var4', data=participant_data
    ).fit(cov_type='HC0')
    model_prob_participant = ols(
        'ln_prob ~ var1 + var2 + var3 + var4', data=participant_data
    ).fit(cov_type='HC0')
    # model_belief_participant.summary()
    participant_betas = model_belief_participant.params.tolist()
    true_betas = model_prob_participant.params.tolist()
    df['coef_var1'].loc[df['label']
                        == participant] = participant_betas[1]
    df['coef_var2'].loc[df['label']
                        == participant] = participant_betas[2]
    df['coef_var3'].loc[df['label']
                        == participant] = participant_betas[3]
    df['coef_var4'].loc[df['label']
                        == participant] = participant_betas[4]
    df['beta_1'].loc[df['label']
                     == participant] = true_betas[1]
    df['beta_2'].loc[df['label']
                     == participant] = true_betas[2]
    df['beta_3'].loc[df['label']
                     == participant] = true_betas[3]
    df['beta_4'].loc[df['label']
                     == participant] = true_betas[4]
    df['diff_var1'].loc[df['label'] ==
                        participant] = (participant_betas[1]-true_betas[1])
    df['diff_var2'].loc[df['label'] ==
                        participant] = (participant_betas[2]-true_betas[2])
    df['diff_var3'].loc[df['label'] ==
                        participant] = (participant_betas[3]-true_betas[3])
    df['diff_var4'].loc[df['label'] ==
                        participant] = (participant_betas[4]-true_betas[4])

# %%
df = df.drop(columns=['level_0', 'index'])
# %%
df[['ln_prob', 'prob', 'var1', 'var2', 'var3', 'var4']
   ].loc[abs(df['diff_var1']) < 0.1]
# %%
# Mann Whitney U test between coefficients
# stat_1, p_1 = mannwhitneyu(df['coef_var1'], df['beta_1'])
# stat_2, p_2 = mannwhitneyu(df['coef_var2'], df['beta_2'])
# stat_3, p_3 = mannwhitneyu(df['coef_var3'], df['beta_3'])
# stat_4, p_4 = mannwhitneyu(df['coef_var4'], df['beta_4'])

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
# how do different treatments affect the coefficient for variable 1,2,3,4?
model_var_1 = ols(
    'coef_var1 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_2 = ols(
    'coef_var2 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_3 = ols(
    'coef_var3 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_4 = ols(
    'coef_var4 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

# how do different treatments affect the difference between the participant beta and true beta for each variable?

model_var_diff_1 = ols(
    'diff_var1 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_diff_2 = ols(
    'diff_var2 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_diff_3 = ols(
    'diff_var3 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

model_var_diff_4 = ols(
    'diff_var4 ~ C(treatment)  + round_number+ loss_amount ', data=df).fit()  # fit(cov_type="HC0")

# %%
os.chdir(base_directory+'ols_s8_to_s17/')

models_coefs_names = [
    'model_var_1', 'model_var_2', 'model_var_3', 'model_var_4',
    'model_var_diff_1', 'model_var_diff_2', 'model_var_diff_3', 'model_var_diff_4'
]

models_coefs = [
    model_var_1, model_var_2, model_var_3, model_var_4,
    model_var_diff_1, model_var_diff_2, model_var_diff_3, model_var_diff_4
]

for m in range(len(models_coefs)):
    with open(models_coefs_names[m] + '.txt', 'w') as f:
        for table in models_coefs[m].summary().tables:
            f.write(
                table.as_latex_tabular()
            )
    f.close()
