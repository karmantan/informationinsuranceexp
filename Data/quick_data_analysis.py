# %%
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
# pwd = "H:\\OneDrive\\Forschung\\Insurance\\Data"
# pwd = 'U:\\documents\\Chapter_2\\Data\\'
# pwd = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/'
# os.chdir(pwd)
# os.chdir('raw/s2/')
# pwd = 'K:\\OneDrive\\Insurance\\Data\\'
# os.chdir(pwd)
# current_folder = "raw\\s10"
# os.chdir(current_folder)
pwd = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/'
os.chdir(pwd)
current_folder = "raw/s17"
os.chdir(current_folder)

data_list = []
# with open(pwd+current_folder+'\\data.csv') as csv_file:
with open(pwd+current_folder+'/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data_list.append(row)

data = pd.DataFrame(data_list[1:], columns=data_list[0])
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
        'HL_switchpoint', 'HLLF_switchpoint']

data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')


# %
# calculate difference between prob and belief
diff_belief_prob = []

for p in range(len(data)):
    diff_belief_prob.append(data.iloc[p]['belief'] - 100*data.iloc[p]['prob'])

data = data.assign(diff_belief_prob=diff_belief_prob)
# %

data_belief_prob_treatment = pd.DataFrame(data.groupby(
    ['treatment']).agg({'diff_belief_prob': 'mean'})).reset_index()
# %
data_wtp_by_loss_amount = pd.DataFrame(data.groupby(
    ['loss_amount']).agg({'WTP': 'mean'})).reset_index()

# %
# import seaborn as sns

data['diff_belief_prob'].hist(by=data['treatment'])
data['WTP'].hist(by=data['loss_amount'])  # %%
# %%
b, m = polyfit(data['belief'], data['WTP'], 1)
plt.plot(data['belief'], data['WTP'], '.')
plt.plot(data['belief'], b+m*data['belief'], '-')
plt.xlabel('Belief')
plt.ylabel('WTP')
plt.show()
# %%
plt.bar(data_belief_prob_treatment['treatment'].tolist(
), data_belief_prob_treatment['diff_belief_prob'].tolist())
plt.xlabel('Treatment')
plt.ylabel('Diff between belief and prob')
plt.axhline(y=0, linewidth=1, color='k')
plt.show()
# %%
plt.bar(data_wtp_by_loss_amount['loss_amount'].tolist(
), data_wtp_by_loss_amount['WTP'].tolist())
plt.xlabel('Loss amount')
plt.ylabel('WTP')
plt.axhline(y=0, linewidth=1, color='k')
plt.show()
# %%
timers = ['timer_char{num}'.format(num=num) for num in range(1, 5)]
# %%
# plot accuracy and time spent scatter plot
all_timers = []
for p in range(len(data)):
    timer_all = []
    for timer in timers:
        timer_all.append(data.iloc[p][timer])
    all_timers.append(sum(timer_all))

data = data.assign(timer_all_chars=all_timers)
data = data.fillna(0)
# %%
# b,m = polyfit(data['timer_all_chars'],data['diff_belief_prob'],1)
plt.plot(data['timer_all_chars'], data['diff_belief_prob'], '.')
# plt.plot(data['timer_all_chars'],b+m*data['timer_all_chars'],'-')
plt.xlabel('Time spent on characteristics')
plt.ylabel('Difference between belief and prob')
plt.show()

# %%
#################################################################
#################################################################
#################################################################
#
#  combine all datasets together
base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'
# base_directory = 'K:\\OneDrive\\Insurance\\Data\\raw\\'
os.chdir(base_directory)
raw_directory = sorted(os.listdir(base_directory))
# %


# %%
last_folder = 17
data_combined = []
for folder in ['s{num}'.format(num=num) for num in range(8, last_folder)]:
    # with open(base_directory+folder+'\\data.csv') as csv_file:
    with open(base_directory+folder+'/data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_combined.append(row)
# %
df = pd.DataFrame(data_combined[1:], columns=data_combined[0])
# %
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
        'HL_switchpoint', 'HLLF_switchpoint']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# %
# calculate difference between prob and belief
diff_belief_prob = []

for p in range(len(df)):
    diff_belief_prob.append(df.iloc[p]['belief'] - 100*df.iloc[p]['prob'])

df = df.assign(diff_belief_prob=diff_belief_prob)

# drop empty rows
df = df.drop(df.loc[df['treatment'] == 'treatment'].index)

# %

data_belief_prob_treatment = pd.DataFrame(df.groupby(
    ['treatment']).agg({'diff_belief_prob': 'mean'})).reset_index()
# %
data_wtp_by_loss_amount = pd.DataFrame(df.groupby(
    ['loss_amount']).agg({'WTP': 'mean'})).reset_index()

# %%
# os.chdir(base_directory+'\\plots_s8_to_s13')

os.chdir(base_directory+'/plots_s8_to_s17')
# %%
plt.bar(data_belief_prob_treatment['treatment'].tolist(
), data_belief_prob_treatment['diff_belief_prob'].tolist())
plt.xlabel('Treatment')
plt.ylabel('Diff between belief and prob')
plt.axhline(y=0, linewidth=1, color='k')
plt.savefig('diff_belief_prob_treatment_bars.png')
plt.show()

# %%
plt.bar(data_wtp_by_loss_amount['loss_amount'].tolist(
), data_wtp_by_loss_amount['WTP'].tolist(), width=15)
plt.xticks(np.arange(min(data_wtp_by_loss_amount['loss_amount']), max(
    data_wtp_by_loss_amount['loss_amount'])+10, 20))
plt.xlabel('Loss amount')
plt.ylabel('WTP')
plt.axhline(y=0, linewidth=1, color='k')
plt.savefig('loss_amount_wtp_bars.png')
plt.show()
# %%
timers = ['timer_char{num}'.format(num=num) for num in range(1, 5)]
# %
# plot accuracy and time spent scatter plot
all_timers = []
for p in range(len(df)):
    timer_all = []
    for timer in timers:
        timer_all.append(df.iloc[p][timer])
    all_timers.append(sum(timer_all))

df = df.assign(timer_all_chars=all_timers)
df['timer_all_chars'] = df['timer_all_chars'].fillna(0)
df['timer_char7'] = df['timer_char7'].fillna(0)
os.chdir(base_directory)
df.to_csv('data_combined_s8_to_s17.csv', index=False)
# %%
# os.chdir(base_directory+'\\plots_s8_to_s10')
os.chdir(base_directory+'/plots_s8_to_s17')
# b,m = polyfit(data['timer_all_chars'],data['diff_belief_prob'],1)
plt.plot(df['timer_all_chars'], df['diff_belief_prob'], '.')
# plt.plot(data['timer_all_chars'],b+m*data['timer_all_chars'],'-')
plt.xlabel('Time spent on characteristics')
plt.ylabel('Difference between belief and prob')
plt.savefig('accuracy_timer.png')
plt.show()

# %%
# exclude outliers
df['timer_all_chars'].describe()
df_no_timer_outlier = df[df['timer_all_chars'].between(6, 150000)]
plt.plot(df_no_timer_outlier['timer_all_chars'],
         df_no_timer_outlier['diff_belief_prob'], '.')
plt.xlabel('Time spent on characteristics')
plt.ylabel('Difference between belief and prob')
plt.savefig('accuracy_timer_no_outlier.png')
# %%

df['diff_belief_prob'].hist(by=df['treatment'])
plt.savefig('accuracy_treatment_histogram.png')

# %%
df['WTP'].hist(by=df['loss_amount'])
plt.savefig('wtp_losses_histogram.png')
# %%
b, m = polyfit(df['belief'], df['WTP'], 1)
plt.plot(df['belief'], df['WTP'], '.')
plt.plot(df['belief'], b+m*df['belief'], '-')
plt.xlabel('Belief')
plt.ylabel('WTP')
plt.savefig('belief_wtp.png')
plt.show()

# %%
# get rid of participants who behaved inconsistently in HL tasks
df_filtered = df.loc[(df['HL_switchpoint'].notnull()) &
                     (df['HLLF_switchpoint'].notnull())]
os.chdir(base_directory)
df_filtered.to_csv('data_filtered_combined_s8_to_s17.csv', index=False)
# %%
os.chdir(base_directory+'filtered_plots_s8_to_s17')

data_belief_prob_treatment = pd.DataFrame(df_filtered.groupby(
    ['treatment']).agg({'diff_belief_prob': 'mean'})).reset_index()
# %
data_wtp_by_loss_amount = pd.DataFrame(df_filtered.groupby(
    ['loss_amount']).agg({'WTP': 'mean'})).reset_index()

plt.bar(data_belief_prob_treatment['treatment'].tolist(
), data_belief_prob_treatment['diff_belief_prob'].tolist())
plt.xlabel('Treatment')
plt.ylabel('Diff between belief and prob')
plt.axhline(y=0, linewidth=1, color='k')
plt.savefig('diff_belief_prob_treatment_bars.png')
plt.show()

plt.bar(data_wtp_by_loss_amount['loss_amount'].tolist(
), data_wtp_by_loss_amount['WTP'].tolist(), width=15)
plt.xticks(np.arange(min(data_wtp_by_loss_amount['loss_amount']), max(
    data_wtp_by_loss_amount['loss_amount'])+10, 20))
plt.xlabel('Loss amount')
plt.ylabel('WTP')
plt.axhline(y=0, linewidth=1, color='k')
plt.savefig('loss_amount_wtp_bars.png')
plt.show()

# b,m = polyfit(data['timer_all_chars'],data['diff_belief_prob'],1)
plt.plot(df_filtered['timer_all_chars'], df_filtered['diff_belief_prob'], '.')
# plt.plot(data['timer_all_chars'],b+m*data['timer_all_chars'],'-')
plt.xlabel('Time spent on characteristics')
plt.ylabel('Difference between belief and prob')
plt.savefig('accuracy_timer.png')
plt.show()

df_filtered['diff_belief_prob'].hist(by=df_filtered['treatment'])
plt.savefig('accuracy_treatment_histogram.png')

df_filtered['timer_all_chars'].describe()
df_filtered_no_timer_outlier = df_filtered[df_filtered['timer_all_chars'].between(
    6, 150000)]
plt.plot(df_filtered_no_timer_outlier['timer_all_chars'],
         df_filtered_no_timer_outlier['diff_belief_prob'], '.')
plt.xlabel('Time spent on characteristics')
plt.ylabel('Difference between belief and prob')
plt.savefig('accuracy_timer_no_outlier.png')

# %
df_filtered['WTP'].hist(by=df_filtered['loss_amount'])
plt.savefig('wtp_losses_histogram.png')
# %
b, m = polyfit(df_filtered['belief'], df_filtered['WTP'], 1)
plt.plot(df_filtered['belief'], df_filtered['WTP'], '.')
plt.plot(df_filtered['belief'], b+m*df_filtered['belief'], '-')
plt.xlabel('Belief')
plt.ylabel('WTP')
plt.savefig('belief_wtp.png')
plt.show()


# %%
##########################################################
##########################################################
##########################################################
# base_directory = 'K:\\OneDrive\\Insurance\\Data\\raw\\'
base_directory = '/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/Data/raw/'

os.chdir(base_directory)

data = []
with open(base_directory+'data_filtered_combined_s8_to_s17.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data.append(row)
# %
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


# %%
plt.plot(df['prob'], df['belief'], '.')
plt.xticks(np.arange(0, 1.05, 0.05), rotation=90)
plt.xlabel('Prob')
plt.ylabel('Belief')
# plt.savefig('filtered_plots_s8_to_s10\\belief_prob.png')
plt.savefig('filtered_plots_s8_to_s17/belief_prob.png')
plt.show()

# %%
b, m = polyfit(df['belief'], df['WTP'], 1)
plt.plot(df['belief'], df['WTP'], '.')
plt.plot(df['belief'], b+m*df['belief'], '-')
plt.xlabel('Belief')
plt.ylabel('WTP')
# plt.savefig('filtered_plots_s8_to_s10\\belief_wtp.png')
plt.savefig('filtered_plots_s8_to_s17/belief_wtp.png')
plt.show()
# %%

# find variance for each probability decile
df['decile'] = pd.qcut(df['prob'], q=10, labels=False)
# %%
data_belief_prob_treatment = pd.DataFrame(df.groupby(['decile']).agg({'diff_belief_prob': ['mean', 'var'],
                                                                      'belief': ['mean', 'var'],
                                                                      'prob': ['mean', 'var']})).reset_index()

# %%
plt.plot(df['prob'], df['diff_belief_prob'], '.')
plt.xticks(np.arange(0, 1.05, .05), rotation=90)
plt.ylabel('Diff between belief and prob')
plt.xlabel('Prob')
plt.savefig('filtered_plots_s8_to_s17/Diff btw belief and prob on prob.png')
# plt.savefig('filtered_plots_s8_to_s10\\Diff btw belief and prob on prob.png')
plt.show()

# %%
############################################################
# # #                NOPE NOT POSSIBLE                 # # #
############################################################

# group by probability and treatment
# check for each probability what do people think their initial probability
# is 'before' receiving information
# ie see what people in the baseline think their probability is.
# baseline = 'before'

for probability in df['prob'].unique().tolist():
    temp_df = df.loc[df['prob'] == probability]
    temp_df_by_treatment = pd.DataFrame(temp_df.groupby(
        ['treatment']).agg({'belief': 'mean'})).reset_index()
    # plt.bar(temp_df_by_treatment['treatment'], temp_df_by_treatment['belief'])
    # plt.xlabel('Treatment, Probability = '+str(probability))
    # plt.ylabel('Belief')
    # plt.savefig('filtered_plots_s8_to_s17/'+str(probability)+'.png')
    # # plt.savefig('filtered_plots_s8_to_s10\\'+str(probability)+'.png')
    # plt.show()


# see difference in probabilities between true probability for each treatment
for probability in df['prob'].unique().tolist():
    temp_df = df.loc[df['prob'] == probability]
    temp_df_by_treatment = pd.DataFrame(temp_df.groupby(
        ['treatment']).agg({'diff_belief_prob': 'mean'})).reset_index()
    # plt.bar(temp_df_by_treatment['treatment'],
    #         temp_df_by_treatment['diff_belief_prob'])
    # plt.xlabel('Treatment, Probability = '+str(probability))
    # plt.ylabel('Diff belief & prob')
    # plt.axhline(y=0, linewidth=1, color='k')
    # plt.savefig('filtered_plots_s8_to_s17/'+str(probability)+'_diff.png')
    # # plt.savefig('filtered_plots_s8_to_s10\\'+str(probability)+'_diff.png')
    # plt.show()
# %%
# for each treatment, see what different probabilities are presented
df['prob'].hist(by=df['treatment'])
# plt.savefig('filtered_plots_s8_to_s10\\prob_treatment_histogram.png')
plt.savefig('filtered_plots_s8_to_s17/prob_treatment_histogram.png')
plt.show()


# %%
# what would the probabilities be if individuals only accounted for the
# positive or negative or variable info

prob_adjusted_for_treatment = []
with open('/Users/karmantan/Library/CloudStorage/OneDrive-Personal/Insurance/treatment_prob_sum.csv') as csv_file:
    # with open('K:\\OneDrive\\Insurance\\treatment_prob_sum.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        prob_adjusted_for_treatment.append(row)

prob_adjusted_for_treatment[0][0] = 'true_prob'
# %%
prob_adj_treat_df = pd.DataFrame(
    prob_adjusted_for_treatment[1:], columns=prob_adjusted_for_treatment[0])
cols = prob_adjusted_for_treatment[0]
prob_adj_treat_df[cols] = prob_adj_treat_df[cols].apply(
    pd.to_numeric, errors='coerce')
# %%

prob_adj_treat_df['prob'] = list(set(df['prob'].tolist()))
# %%
df = pd.merge(df, prob_adj_treat_df, how='left', on='prob').reset_index()
# %%
df.drop(columns=['index'])
# %%
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

# %%
plt.bar(df_adj_belief_prob_treatment['treatment'].tolist(
), df_adj_belief_prob_treatment['diff_belief_prob_adj_treatment'].tolist())
plt.xlabel('Treatment')
plt.ylabel('Diff between belief and prob, adjusted')
plt.axhline(y=0, linewidth=1, color='k')
plt.savefig(
    'filtered_plots_s8_to_s17/diff_belief_prob_treatment_bars_adjusted.png')
# plt.savefig(
# 'filtered_plots_s8_to_s10\\diff_belief_prob_treatment_bars_adjusted.png')
plt.show()

# %%
model_0_accuracy = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob', data=df).fit(cov_type="HC0")


"""
fullinfo: accuracy decreases by a lot, significant 1%
neginfo: accuracy decreases, significant 1%
posinfo: accuracy decreases, significant 1%
varinfo: accuracy increases, significant 1%
"""
# %
model_0_accuracy.summary()

model_1_accuracy = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob + timer_all_chars + timer_char7 + round_number', data=df).fit(cov_type="HC0")


"""
fullinfo: accuracy decreases by a lot, significant 1%
neginfo: accuracy decreases, significant 1%
posinfo: accuracy decreases, significant 1%
varinfo: accuracy increases, significant 1%
"""
# %
model_1_accuracy.summary()
# %
model_2_accuracy = ols(
    'diff_belief_prob ~ C(treatment) + prob + timer_all_chars + timer_char7 + round_number', data=df).fit(cov_type="HC0")

"""
fullinfo: insignificant
neginfo: underestimate probabilities, significant 1%
posinfo: underestimate probabilities, significant 1%
varinfo: underestimate probabilities, significant 10%
prob: the higher the prob, the lower the diff btw belief and prob, sig 1%
in other words, for low probabilities diff btw belief and prob positive (overestimation).
for high probabilities, diff btw belief and prob negative (underestimation)
"""
# %
model_2_accuracy.summary()

# %
model_3_accuracy = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob + loss_amount + timer_all_chars + timer_char7 + round_number', data=df).fit(cov_type="HC0")

"""
fullinfo: accuracy decreases by a lot, significant 1%
neginfo: accuracy decreases, significant 1%
posinfo: accuracy decreases, significant 1%
varinfo: accuracy increases, significant 1%
"""
# %
model_3_accuracy.summary()

# %
model_4_accuracy = ols(
    'diff_belief_prob ~ C(treatment) + prob + loss_amount + timer_all_chars + timer_char7 + round_number', data=df).fit(cov_type="HC0")

"""
fullinfo: insignificant
neginfo: underestimate probabilities, significant 1%
posinfo: underestimate probabilities, significant 1%
varinfo: overestimate probabilities, significant 1%
prob: the higher the probability, the lower the difference in beliefs, sig 1%
"""
# %
model_4_accuracy.summary()

# %
model_5_belief = ols(
    'belief ~ C(treatment) + prob + loss_amount + timer_all_chars + timer_char7 + round_number', data=df).fit(cov_type="HC0")

"""
Providing more information than baseline leads people
to decrease their belief of suffering a loss
except in varinfo case (belief increased)
"""
# %
model_5_belief.summary()

# %
model_6_wtp = ols('WTP ~ belief + number_entered + loss_amount+ round_number',
                  data=df).fit(cov_type="HC0")
model_6_wtp.summary()
# %
model_7_wtp = ols(
    'WTP ~ belief + number_entered + loss_amount + C(treatment)+ round_number', data=df).fit(cov_type="HC0")
model_7_wtp.summary()
# %
model_8_accuracy = ols('belief ~ var1 + var2 + var3 + var4+ round_number',
                       data=df).fit(cov_type="HC0")

"""
variables affect risk the way they should
"""
# %
model_8_accuracy.summary()

# %
model_9_compare_betas = ols(
    'prob ~ var1 + var2 + var3 + var4+ round_number', data=df).fit(cov_type="HC0")

"""
var1: people overestimate impact of var1 on risk-increasing capability
var2: people overestimate impact of var2 on risk-increasing capability
var3: people overestimate impact of var3 on risk-decreasing capability
var4: people overestimate impact of var4 on risk-decreasing capability
"""
# %
model_9_compare_betas.summary()

# %%
os.chdir(base_directory+'ols_s8_to_s17/')

model_names = ['model_0_accuracy',
               'model_1_accuracy',
               'model_2_accuracy',
               'model_3_accuracy',
               'model_4_accuracy',
               'model_5_belief',
               'model_6_wtp',
               'model_7_wtp',
               'moodel_8_accuracy',
               'model_9_compare_betas']

model_summaries = [model_0_accuracy.summary(),
                   model_1_accuracy.summary(),
                   model_2_accuracy.summary(),
                   model_3_accuracy.summary(),
                   model_4_accuracy.summary(),
                   model_5_belief.summary(),
                   model_6_wtp.summary(),
                   model_7_wtp.summary(),
                   model_8_accuracy.summary(),
                   model_9_compare_betas.summary()]

for i in range(len(model_summaries)):
    with open(model_names[i] + '.html', 'w') as fh:
        sum_table = model_summaries[i].tables[0].as_html(
        ) + '\n' + model_summaries[i].tables[1].as_html() + '\n' + model_summaries[i].tables[2].as_html()
        fh.write(sum_table)
# %%

# find variance by treatment
var_by_treatment_df = pd.DataFrame(df.groupby(['treatment']).agg({'diff_belief_prob': ['mean', 'var', 'count'],
                                                                  'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                                  'belief': ['mean', 'var'],
                                                                  'prob': ['mean', 'var']})).reset_index()


# %%

# variance by true probability
var_by_prob_df = pd.DataFrame(df.groupby(['prob']).agg({'diff_belief_prob': ['mean', 'var'],
                                                        'diff_belief_prob_adj_treatment': ['mean', 'var'],
                                                        'belief': ['mean', 'var'],
                                                        'prob': ['mean', 'var']})).reset_index()
# %%
print(str(6.3227/0.1441) + ';' + str(1.3897/0.0469) +
      ';' + str(3.0825/0.1067) + ';' + str(1.4568/0.0618))
# %%
demo_info = df[['treatment', 'age', 'gender', 'smoker', 'health', 'exercisedays',
                'exercisehours', 'insurancehealth', 'lifeinsurance', 'riskdriving',
                'riskfinance', 'risksport', 'riskjob', 'riskhealth', 'risk']]

# %%
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
# %%
group_baseline = demo_info.loc[demo_info['treatment'] == 'baseline']
group_fullinfo = demo_info.loc[demo_info['treatment'] == 'fullinfo']
group_neginfo = demo_info.loc[demo_info['treatment'] == 'neginfo']
group_posinfo = demo_info.loc[demo_info['treatment'] == 'posinfo']
group_varinfo = demo_info.loc[demo_info['treatment'] == 'varinfo']
# %%
demo_varnames = ['age', 'gender', 'smoker', 'health', 'exercisedays',
                 'exercisehours', 'insurancehealth', 'lifeinsurance', 'riskdriving',
                 'riskfinance', 'risksport', 'riskjob', 'riskhealth', 'risk']

# %%
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
# %%
tteststats_all_vars = groups_tteststat[1:]
# %%
model_0_accuracy_var = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

# model_1_accuracy_var = ols(
#     'abs(diff_belief_prob) ~ C(treatment) + prob + timer_all_chars + timer_char7 + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

# get rid of multicollinearity by removing timers
model_1_accuracy_var = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

model_2_accuracy_var = ols(
    'diff_belief_prob ~ C(treatment) + prob + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

model_3_accuracy_var = ols(
    'abs(diff_belief_prob) ~ C(treatment) + prob + loss_amount + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

model_4_accuracy_var = ols(
    'diff_belief_prob ~ C(treatment) + loss_amount + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

model_5_belief_var = ols(
    'belief ~ C(treatment) + loss_amount + round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")

model_6_wtp_var = ols('WTP ~ belief + number_entered + loss_amount+ round_number + var1 + var2 + var3 + var4',
                  data=df).fit(cov_type="HC0")

model_7_wtp_var = ols(
    'WTP ~ belief + number_entered + loss_amount + C(treatment)+ round_number + var1 + var2 + var3 + var4', data=df).fit(cov_type="HC0")
# %%
os.chdir(base_directory+'ols_s8_to_s17/')

model_names_var = ['model_0_accuracy_var',
               'model_1_accuracy_var',
               'model_2_accuracy_var',
               'model_3_accuracy_var',
               'model_4_accuracy_var',
               'model_5_belief_var',
               'model_6_wtp_var',
               'model_7_wtp_var',]

model_summaries_var = [model_0_accuracy_var.summary(),
                   model_1_accuracy_var.summary(),
                   model_2_accuracy_var.summary(),
                   model_3_accuracy_var.summary(),
                   model_4_accuracy_var.summary(),
                   model_5_belief_var.summary(),
                   model_6_wtp_var.summary(),
                   model_7_wtp_var.summary(),]

for i in range(len(model_summaries_var)):
    with open(model_names_var[i] + '.html', 'w') as fh:
        sum_table = model_summaries_var[i].tables[0].as_html(
        ) + '\n' + model_summaries_var[i].tables[1].as_html() + '\n' + model_summaries_var[i].tables[2].as_html()
        fh.write(sum_table)
# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%

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
 
df_quantifiable = df.loc[:,features_of_interest]

df_quantifiable[features_of_interest] = df_quantifiable[features_of_interest].apply(pd.to_numeric, errors='coerce')
df_quantifiable.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
#%%
vif = pd.DataFrame()
vif['Features'] = df_quantifiable.columns
vif['VIF'] = [variance_inflation_factor(df_quantifiable.values, i) for i in range(df_quantifiable.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# %%
model_posinfo = ols(
    'abs(diff_belief_prob) ~ prob + loss_amount + round + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='posinfo']).fit(cov_type="HC0")

model_neginfo = ols(
    'abs(diff_belief_prob) ~ prob + loss_amount + round + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='neginfo']).fit(cov_type="HC0")

model_varinfo = ols(
    'abs(diff_belief_prob) ~ prob + loss_amount + round + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='varinfo']).fit(cov_type="HC0")

model_fullinfo = ols(
    'abs(diff_belief_prob) ~ prob + loss_amount + round + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='fullinfo']).fit(cov_type="HC0")

model_baseline = ols(
    'abs(diff_belief_prob) ~ prob + loss_amount + round + var1 + var2 + var3 + var4', data=df.loc[df['treatment']=='baseline']).fit(cov_type="HC0")

os.chdir(base_directory+'ols_s8_to_s17/')

model_names_var_treatment = ['model_posinfo',
                   'model_neginfo',
                   'model_varinfo',
                   'model_fullinfo',
                   'model_baseline']

model_summaries_var_treatment = [model_posinfo.summary(),
                       model_neginfo.summary(),
                       model_varinfo.summary(),
                       model_fullinfo.summary(),
                       model_baseline.summary(),]

for i in range(len(model_summaries_var_treatment)):
    with open(model_names_var_treatment[i] + '.html', 'w') as fh:
        sum_table = model_summaries_var_treatment[i].tables[0].as_html(
        ) + '\n' + model_summaries_var_treatment[i].tables[1].as_html() + '\n' + model_summaries_var_treatment[i].tables[2].as_html()
        fh.write(sum_table)