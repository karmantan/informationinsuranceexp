# %%
import pandas as pd
import numpy as np
import os
import random
import glob
import re

# %
# pwd = "H:\\OneDrive\\Forschung\\Insurance\\Data"
pwd = 'U:\\documents\\Chapter_2\\Data\\'
os.chdir(pwd)
os.chdir("raw\\s2")

prol_files = glob.glob('prolific_*.csv')
prol_dat = pd.DataFrame()
for file in prol_files:
    temp = pd.read_csv(file)
    prol_dat = prol_dat.append(temp)

prol_dat = prol_dat.loc[prol_dat['Status'] == 'APPROVED'].copy()


ot_files = glob.glob('all_apps_wide*.csv')
ot_dat = pd.DataFrame()
for file in ot_files:
    temp = pd.read_csv(file)
    ot_dat = ot_dat.append(temp)

ot_dat = ot_dat.loc[ot_dat['participant.label'].isin(
    prol_dat['Participant id'])]

bonus_payments = ot_dat[['participant.label',
                         'informationtreatment.5.player.display_payoff_euro']].copy()
bonus_payments['informationtreatment.5.player.display_payoff_euro'] = bonus_payments[
    'informationtreatment.5.player.display_payoff_euro'].astype(float)
bonus_payments.to_csv('bonus_payments.csv', sep=",",
                      float_format='%.2f', index=False)
# %%
# print(ot_dat.columns.tolist())

rel_cols = ['informationtreatment.1.subsession.round_number', 'informationtreatment.2.subsession.round_number', 'informationtreatment.3.subsession.round_number', 'informationtreatment.4.subsession.round_number', 'informationtreatment.5.subsession.round_number', 'participant.label',  'participant.treatment',  'session.code',  'preexperiment.1.player.questionnaire_age',  'preexperiment.1.player.questionnaire_gender',  'preexperiment.1.player.questionnaire_highestqualification',  'preexperiment.1.player.questionnaire_study',  'preexperiment.1.player.questionnaire_semester',  'preexperiment.1.player.questionnaire_smoker',  'preexperiment.1.player.questionnaire_health',  'preexperiment.1.player.questionnaire_exercisedays',  'preexperiment.1.player.questionnaire_exercisehours',  'preexperiment.1.player.questionnaire_insurancehealth',  'preexperiment.1.player.questionnaire_parentsplan',  'preexperiment.1.player.questionnaire_haftpflicht',  'preexperiment.1.player.questionnaire_carinsurance',  'preexperiment.1.player.questionnaire_disabilityinsurance',  'preexperiment.1.player.questionnaire_lifeinsurance',  'preexperiment.1.player.questionnaire_hausrat',  'preexperiment.1.player.questionnaire_legalinsurance',  'preexperiment.1.player.questionnaire_riskdriving',  'preexperiment.1.player.questionnaire_riskfinance',  'preexperiment.1.player.questionnaire_risksport',  'preexperiment.1.player.questionnaire_riskjob',  'preexperiment.1.player.questionnaire_riskhealth',  'preexperiment.1.player.questionnaire_risktrust',  'preexperiment.1.player.questionnaire_risk',  'preexperiment.1.player.number_entered',  'preexperiment.1.player.result_risk',  'preexperiment.1.player.choice1',  'preexperiment.1.player.choice2',  'preexperiment.1.player.choice3',  'preexperiment.1.player.choice4',  'preexperiment.1.player.choice5',  'preexperiment.1.player.choice6',  'preexperiment.1.player.choice7',  'preexperiment.1.player.choice8',  'preexperiment.1.player.choice9',  'preexperiment.1.player.choice10',  'preexperiment.1.player.choice1LF',  'preexperiment.1.player.choice2LF',  'preexperiment.1.player.choice3LF',  'preexperiment.1.player.choice4LF',  'preexperiment.1.player.choice5LF',  'preexperiment.1.player.choice6LF',  'preexperiment.1.player.choice7LF',  'preexperiment.1.player.choice8LF',  'preexperiment.1.player.choice9LF',  'preexperiment.1.player.choice10LF',  'informationtreatment.1.player.var1',  'informationtreatment.1.player.var2',  'informationtreatment.1.player.var3',  'informationtreatment.1.player.var4',  'informationtreatment.1.player.var5',  'informationtreatment.1.player.var6',  'informationtreatment.1.player.prob',  'informationtreatment.1.player.belief',  'informationtreatment.1.player.WTP',  'informationtreatment.1.player.insurance',  'informationtreatment.1.player.timer_char1',  'informationtreatment.1.player.timer_char2',  'informationtreatment.1.player.timer_char3',  'informationtreatment.1.player.timer_char4',  'informationtreatment.1.player.timer_char5',  'informationtreatment.1.player.timer_char6',  'informationtreatment.1.player.timer_faq_coverage',  'informationtreatment.1.player.timer_faq_noinsurance',  'informationtreatment.1.player.timer_faq_howitworks',  'informationtreatment.1.player.timer_faq_choiceaffectsprice',  'informationtreatment.1.player.state',  'informationtreatment.1.player.random_number',  'informationtreatment.1.player.insurance_purchase',  'informationtreatment.1.player.loss_amount',  'informationtreatment.2.player.var1',  'informationtreatment.2.player.var2',  'informationtreatment.2.player.var3',  'informationtreatment.2.player.var4',  'informationtreatment.2.player.var5',  'informationtreatment.2.player.var6',  'informationtreatment.2.player.prob',  'informationtreatment.2.player.belief',  'informationtreatment.2.player.WTP',  'informationtreatment.2.player.insurance',  'informationtreatment.2.player.timer_char1',
            'informationtreatment.2.player.timer_char2',  'informationtreatment.2.player.timer_char3',  'informationtreatment.2.player.timer_char4',  'informationtreatment.2.player.timer_char5',  'informationtreatment.2.player.timer_char6',  'informationtreatment.2.player.timer_faq_coverage',  'informationtreatment.2.player.timer_faq_noinsurance',  'informationtreatment.2.player.timer_faq_howitworks',  'informationtreatment.2.player.timer_faq_choiceaffectsprice',  'informationtreatment.2.player.state',  'informationtreatment.2.player.random_number',  'informationtreatment.2.player.insurance_purchase',  'informationtreatment.2.player.loss_amount',  'informationtreatment.3.player.var1',  'informationtreatment.3.player.var2',  'informationtreatment.3.player.var3',  'informationtreatment.3.player.var4',  'informationtreatment.3.player.var5',  'informationtreatment.3.player.var6',  'informationtreatment.3.player.prob',  'informationtreatment.3.player.belief',  'informationtreatment.3.player.WTP',  'informationtreatment.3.player.insurance',  'informationtreatment.3.player.timer_char1',  'informationtreatment.3.player.timer_char2',  'informationtreatment.3.player.timer_char3',  'informationtreatment.3.player.timer_char4',  'informationtreatment.3.player.timer_char5',  'informationtreatment.3.player.timer_char6',  'informationtreatment.3.player.timer_faq_coverage',  'informationtreatment.3.player.timer_faq_noinsurance',  'informationtreatment.3.player.timer_faq_howitworks',  'informationtreatment.3.player.timer_faq_choiceaffectsprice',  'informationtreatment.3.player.state',  'informationtreatment.3.player.random_number',  'informationtreatment.3.player.insurance_purchase',  'informationtreatment.3.player.loss_amount',  'informationtreatment.4.player.var1',  'informationtreatment.4.player.var2',  'informationtreatment.4.player.var3',  'informationtreatment.4.player.var4',  'informationtreatment.4.player.var5',  'informationtreatment.4.player.var6',  'informationtreatment.4.player.prob',  'informationtreatment.4.player.belief',  'informationtreatment.4.player.WTP',  'informationtreatment.4.player.insurance',  'informationtreatment.4.player.timer_char1',  'informationtreatment.4.player.timer_char2',  'informationtreatment.4.player.timer_char3',  'informationtreatment.4.player.timer_char4',  'informationtreatment.4.player.timer_char5',  'informationtreatment.4.player.timer_char6',  'informationtreatment.4.player.timer_faq_coverage',  'informationtreatment.4.player.timer_faq_noinsurance',  'informationtreatment.4.player.timer_faq_howitworks',  'informationtreatment.4.player.timer_faq_choiceaffectsprice',  'informationtreatment.4.player.state',  'informationtreatment.4.player.random_number',  'informationtreatment.4.player.insurance_purchase',  'informationtreatment.4.player.loss_amount',  'informationtreatment.5.player.var1',  'informationtreatment.5.player.var2',  'informationtreatment.5.player.var3',  'informationtreatment.5.player.var4',  'informationtreatment.5.player.var5',  'informationtreatment.5.player.var6',  'informationtreatment.5.player.prob',  'informationtreatment.5.player.belief',  'informationtreatment.5.player.WTP',  'informationtreatment.5.player.insurance',  'informationtreatment.5.player.timer_char1',  'informationtreatment.5.player.timer_char2',  'informationtreatment.5.player.timer_char3',  'informationtreatment.5.player.timer_char4',  'informationtreatment.5.player.timer_char5',  'informationtreatment.5.player.timer_char6',  'informationtreatment.5.player.timer_faq_coverage',  'informationtreatment.5.player.timer_faq_noinsurance',  'informationtreatment.5.player.timer_faq_howitworks',  'informationtreatment.5.player.timer_faq_choiceaffectsprice',  'informationtreatment.5.player.state',  'informationtreatment.5.player.random_number',  'informationtreatment.5.player.insurance_purchase',  'informationtreatment.5.player.final_payoff',  'informationtreatment.5.player.loss_amount']
oneround_cols = ['participant.label',  'participant.treatment',  'session.code',  'preexperiment.1.player.questionnaire_age',  'preexperiment.1.player.questionnaire_gender',  'preexperiment.1.player.questionnaire_highestqualification',  'preexperiment.1.player.questionnaire_study',  'preexperiment.1.player.questionnaire_semester',  'preexperiment.1.player.questionnaire_smoker',  'preexperiment.1.player.questionnaire_health',  'preexperiment.1.player.questionnaire_exercisedays',  'preexperiment.1.player.questionnaire_exercisehours',  'preexperiment.1.player.questionnaire_insurancehealth',  'preexperiment.1.player.questionnaire_parentsplan',  'preexperiment.1.player.questionnaire_haftpflicht',  'preexperiment.1.player.questionnaire_carinsurance',  'preexperiment.1.player.questionnaire_disabilityinsurance',  'preexperiment.1.player.questionnaire_lifeinsurance',  'preexperiment.1.player.questionnaire_hausrat',  'preexperiment.1.player.questionnaire_legalinsurance',  'preexperiment.1.player.questionnaire_riskdriving',  'preexperiment.1.player.questionnaire_riskfinance',
                 'preexperiment.1.player.questionnaire_risksport',  'preexperiment.1.player.questionnaire_riskjob',  'preexperiment.1.player.questionnaire_riskhealth',  'preexperiment.1.player.questionnaire_risktrust',  'preexperiment.1.player.questionnaire_risk',  'preexperiment.1.player.number_entered',  'preexperiment.1.player.result_risk',  'preexperiment.1.player.choice1',  'preexperiment.1.player.choice2',  'preexperiment.1.player.choice3',  'preexperiment.1.player.choice4',  'preexperiment.1.player.choice5',  'preexperiment.1.player.choice6',  'preexperiment.1.player.choice7',  'preexperiment.1.player.choice8',  'preexperiment.1.player.choice9',  'preexperiment.1.player.choice10',  'preexperiment.1.player.choice1LF',  'preexperiment.1.player.choice2LF',  'preexperiment.1.player.choice3LF',  'preexperiment.1.player.choice4LF',  'preexperiment.1.player.choice5LF',  'preexperiment.1.player.choice6LF',  'preexperiment.1.player.choice7LF',  'preexperiment.1.player.choice8LF',  'preexperiment.1.player.choice9LF',  'preexperiment.1.player.choice10LF', 'informationtreatment.5.player.final_payoff']
ot_dat = ot_dat[rel_cols].copy()
data = ot_dat[oneround_cols].copy()
ot_dat.rename(columns={'informationtreatment.1.player.insurance': 'informationtreatment.1.player.insurancex',  'informationtreatment.2.player.insurance': 'informationtreatment.2.player.insurancex',  'informationtreatment.3.player.insurance':
              'informationtreatment.3.player.insurancex',  'informationtreatment.4.player.insurance': 'informationtreatment.4.player.insurancex',  'informationtreatment.5.player.insurance': 'informationtreatment.5.player.insurancex'}, inplace=True)


# %
vars = ot_dat.columns.to_list()


var = 'informationtreatment.*.subsession.round_number'
var_name = var.split(".")[-1]
oldvar_name = '~'+var_name
regex = re.compile(var)
current_vars = [v for v in vars if re.match(regex, v)]
temp = pd.melt(ot_dat, id_vars=['participant.label'],
               value_vars=current_vars,
               value_name=var_name,
               var_name=oldvar_name)
data = data.merge(temp[['participant.label', var_name]],
                  on='participant.label')

#just_lab_rounds = ot_dat[['participant.label', 'round_number']].copy()

# %

# print(ot_dat.columns.tolist())
data_vars = ['informationtreatment.*.insurancex', 'informationtreatment.*.var1', 'informationtreatment.*.var2', 'informationtreatment.*.var3', 'informationtreatment.*.var4', 'informationtreatment.*.var5', 'informationtreatment.*.var6', 'informationtreatment.*.prob', 'informationtreatment.*.belief', 'informationtreatment.*.WTP',  'informationtreatment.*.timer_char1', 'informationtreatment.*.timer_char2', 'informationtreatment.*.timer_char3',
             'informationtreatment.*.timer_char4', 'informationtreatment.*.timer_char5', 'informationtreatment.*.timer_char6', 'informationtreatment.*.timer_faq_coverage', 'informationtreatment.*.timer_faq_noinsurance', 'informationtreatment.*.timer_faq_howitworks', 'informationtreatment.*.timer_faq_choiceaffectsprice', 'informationtreatment.*.state', 'informationtreatment.*.random_number',  'informationtreatment.*.loss_amount']

for var in data_vars:
    var_name = var.split(".")[-1]
    oldvar_name = '~'+var_name
    regex = re.compile(var)
    current_vars = [v for v in vars if re.match(regex, v)]
    temp = pd.melt(ot_dat, id_vars=['participant.label'],
                   value_vars=current_vars,
                   value_name=var_name,
                   var_name=oldvar_name)
    temp['round_number'] = temp[oldvar_name].str.extract('(\d+)').astype(int)
    data = data.merge(temp[['participant.label', var_name, 'round_number']], on=[
                      'participant.label', 'round_number'])


# %

data.columns = data.columns.str.replace('informationtreatment.5.player', '')
data.columns = data.columns.str.replace('player.', '')
data.columns = data.columns.str.replace('preexperiment.1.', '')
data.columns = data.columns.str.replace('questionnaire_', '')
data.columns = data.columns.str.replace('participant.', '')

# condense HL
choices_HL_var_names = ['choice{num}'.format(num=num) for num in range(1, 11)]
choices_HLLF_var_names = ['choice{num}LF'.format(
    num=num) for num in range(1, 11)]

switchpoints = []
for p in range(len(data)):
    i = 0
    while list(dict(data.iloc[p][choices_HL_var_names]).values())[i] == 1 and i < 9:
        i += 1
    else:
        j = i + 1
        if 1.0 in list(dict(data.iloc[p][choices_HL_var_names]).values())[j:]:
            switchpoints.append(None)
        else:
            switchpoints.append(i)

data = data.assign(HL_switchpoint=switchpoints)

switchpoints_LF = []
for p in range(len(data)):
    i = 0
    while list(dict(data.iloc[p][choices_HLLF_var_names]).values())[i] == 1 and i < 9:
        i += 1
    else:
        j = i + 1
        if 1.0 in list(dict(data.iloc[p][choices_HLLF_var_names]).values())[j:]:
            switchpoints_LF.append(None)
        else:
            switchpoints_LF.append(i)


data = data.assign(HLLF_switchpoint=switchpoints_LF)

# drop hl and hllf columns
data = data.drop(columns=choices_HL_var_names+choices_HLLF_var_names)

# %%
os.chdir(pwd)
data.to_csv('data.csv', index=False)
