from otree.api import *
import random
from settings import LANGUAGE_CODE
doc = """
Your app description
"""

if LANGUAGE_CODE == 'de':
    from preexperiment.lexicon_de import Lexicon
else:
    from preexperiment.lexicon_en import Lexicon

which_language = {'en': False, 'de': False, 'zh': False}  # noqa
which_language[LANGUAGE_CODE[:2]] = True


class C(BaseConstants):
    NAME_IN_URL = 'preexperiment'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    no_of_rounds = 5
    baseline_prob = 0.5
    factor = 2.5
    probability_risk = 1 / 3
    initial = 50
    max_bet = 50
    HLoutcomes = [[1, 16], [38, 20]]
    HLLFoutcomes = [[39, 24], [2, 20]]
    A_win = 20
    A_lose = 16
    B_win = 38
    B_lose = 1
    A_win_loss = 20
    A_lose_loss = 24
    B_win_loss = 2
    B_lose_loss = 39
    HLLFinitial = 40
    treatments = ['baseline', 'fullinfo', 'varinfo', 'neginfo', 'posinfo']
    survey_reward = 50
    conversion = 100
    loss_1 = 20
    loss_2 = 40
    loss_3 = 60
    loss_4 = 80
    loss_5 = 100


def make_booleanfield():
    return models.BooleanField(
        choices=[[True, 'A'], [False, 'B'], ],
        widget=widgets.RadioSelectHorizontal,
    )


def gender_integerfield():
    return models.IntegerField(
        choices=[[1, 'Female'], [0, 'Male'], [
            2, 'Diverse'], [-1, 'Prefer not to say']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def gender_integerfieldde():
    return models.IntegerField(
        choices=[[1, 'weiblich'], [0, 'männlich'], [
            2, 'divers'], [-1, 'Lieber nicht sagen']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def yesno_integerfield():
    return models.IntegerField(
        choices=[[1, 'Yes'], [0, 'No'], [-1, 'Prefer not to say'], ],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def yesno_integerfieldde():
    return models.IntegerField(
        choices=[[1, 'Ja'], [0, 'Nein'], [-1, 'Lieber nicht sagen'], ],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def study_integerfield():
    return models.IntegerField(
        choices=[[1, 'FB01: Rechtswissenschaft'],
                 [2, 'FB02: Wirtschaftswisseenschaften'],
                 [3, 'FB03: Gesellschaftswissenschaften'],
                 [4, 'FB04: Erziehungswissenschaften'],
                 [5, 'FB05: Psychologie und Sportwissenschaften'],
                 [6, 'FB06: Evangelische Theologie'],
                 [7, 'FB07: Katholische Theologie'],
                 [8, 'FB08: Philosophie und Geschichtswissenschaften'],
                 [9, 'FB09: Sprach- und Kulturwissenschaften'],
                 [10, 'FB10: Neuere Philologien'],
                 [11, 'FB11: Geowissenschaften/Geologie'],
                 [12, 'FB12: Informatik und Mathematik'],
                 [13, 'FB13: Physik'],
                 [14, 'FB14: Biochemie, Chemie und Pharmazie'],
                 [15, 'FB15: Biowissenschaften'],
                 [16, 'FB16: Medizin'],
                 [17, 'Not a student'],
                 [-1, 'Prefer not to say']],
        label="",
    )


def study_integerfieldde():
    return models.IntegerField(
        choices=[[1, 'FB01: Rechtswissenschaft'],
                 [2, 'FB02: Wirtschaftswisseenschaften'],
                 [3, 'FB03: Gesellschaftswissenschaften'],
                 [4, 'FB04: Erziehungswissenschaften'],
                 [5, 'FB05: Psychologie und Sportwissenschaften'],
                 [6, 'FB06: Evangelische Theologie'],
                 [7, 'FB07: Katholische Theologie'],
                 [8, 'FB08: Philosophie und Geschichtswissenschaften'],
                 [9, 'FB09: Sprach- und Kulturwissenschaften'],
                 [10, 'FB10: Neuere Philologien'],
                 [11, 'FB11: Geowissenschaften/Geologie'],
                 [12, 'FB12: Informatik und Mathematik'],
                 [13, 'FB13: Physik'],
                 [14, 'FB14: Biochemie, Chemie und Pharmazie'],
                 [15, 'FB15: Biowissenschaften'],
                 [16, 'FB16: Medizin'],
                 [17, 'kein Student'],
                 [-1, 'Lieber nicht sagen']],
        label="",
    )


def ten_integerfield():
    return models.IntegerField(
        choices=[
            [0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5'],
            [6, '6'], [7, '7'], [8, '8'], [9, '9'], [10, '10'], [-1, 'Prefer not to say']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def ten_integerfieldde():
    return models.IntegerField(
        choices=[
            [0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5'],
            [6, '6'], [7, '7'], [8, '8'], [9, '9'], [10, '10'], [-1, 'Lieber nicht sagen']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def seven_integerfield():
    return models.IntegerField(
        choices=[
            [0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5'],
            [6, '6'], [7, '7'], [-1, 'Prefer not to say']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def seven_integerfieldde():
    return models.IntegerField(
        choices=[
            [0, '0'], [1, '1'], [2, '2'], [3, '3'], [4, '4'], [5, '5'],
            [6, '6'], [7, '7'], [-1, 'Lieber nicht sagen']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def highestqualification_integerfield():
    return models.IntegerField(
        choices=[
            [0, 'High School'],
            [1, "Bachelor's"],
            [2, "Master's"],
            [3, "PhD"],
            [4, "Other"], [-1, "Prefer not to say"]],
        label="",
    )


def highestqualification_integerfieldde():
    return models.IntegerField(
        choices=[
            [0, 'Abitur'],
            [1, "Bachelor"],
            [2, "Master"],
            [3, "Promotion"],
            [4, "Sonstige"], [-1, "Lieber nicht sagen"]],
        label="",
    )


def healthinsurance_integerfield():
    return models.IntegerField(
        choices=[
            [1, 'Public insurance'], [0, 'Private insurance'],
            [-1, 'Prefer not to say']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def healthinsurance_integerfieldde():
    return models.IntegerField(
        choices=[
            [1, 'gesetzliche Krankenversicherung'], [
                0, 'private Krankenversicherung'],
            [-1, 'Lieber nicht sagen']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def gps_integerfield():
    return models.IntegerField(
        choices=[[0, '50/50 Chance'], [1, 'Sure payment'],
                 [-1, 'Prefer not to say']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )


def gps_integerfieldde():
    return models.IntegerField(
        choices=[[0, '50/50 Chance'], [1, 'sichere Bezahlung'],
                 [-1, 'Lieber nicht sagen']],
        # widget=widgets.RadioSelectHorizontal,
        label="",
    )



class Player(BasePlayer):
    language_picked = models.StringField()
    questionnaire_age = models.FloatField(min=0, max=100, label="")
    questionnaire_gender = gender_integerfield()
    questionnaire_genderde = gender_integerfieldde()
    questionnaire_smoker = yesno_integerfield()
    questionnaire_smokerde = yesno_integerfieldde()
    questionnaire_health = ten_integerfield()
    questionnaire_healthde = ten_integerfieldde()
    questionnaire_exercisedays = seven_integerfield()
    questionnaire_exercisedaysde = seven_integerfieldde()
    questionnaire_exercisehours = models.FloatField(min=0, max=24, label="")
    questionnaire_insurancehealth = healthinsurance_integerfield()
    questionnaire_insurancehealthde = healthinsurance_integerfieldde()
    questionnaire_lifeinsurance = yesno_integerfield()
    questionnaire_lifeinsurancede = yesno_integerfieldde()
    questionnaire_riskdriving = ten_integerfield()
    questionnaire_riskdrivingde = ten_integerfieldde()
    questionnaire_riskfinance = ten_integerfield()
    questionnaire_riskfinancede = ten_integerfieldde()
    questionnaire_risksport = ten_integerfield()
    questionnaire_risksportde = ten_integerfieldde()
    questionnaire_riskjob = ten_integerfield()
    questionnaire_riskjobde = ten_integerfieldde()
    questionnaire_riskhealth = ten_integerfield()
    questionnaire_riskhealthde = ten_integerfieldde()
    questionnaire_risktrust = ten_integerfield()
    questionnaire_risktrustde = ten_integerfieldde()
    questionnaire_risk = ten_integerfield()
    questionnaire_riskde = ten_integerfieldde()
    number_entered = models.FloatField(min=0, max=C.max_bet, label="")
    result_risk = models.FloatField()
    win_risk = models.BooleanField()
    winHL = models.BooleanField()
    winHLLF = models.BooleanField()
    choiceHL = models.BooleanField()
    choiceHLLF = models.BooleanField()
    choice1 = make_booleanfield()
    choice2 = make_booleanfield()
    choice3 = make_booleanfield()
    choice4 = make_booleanfield()
    choice5 = make_booleanfield()
    choice6 = make_booleanfield()
    choice7 = make_booleanfield()
    choice8 = make_booleanfield()
    choice9 = make_booleanfield()
    choice10 = make_booleanfield()
    winHL1 = models.BooleanField()
    winHL2 = models.BooleanField()
    winHL3 = models.BooleanField()
    winHL4 = models.BooleanField()
    winHL5 = models.BooleanField()
    winHL6 = models.BooleanField()
    winHL7 = models.BooleanField()
    winHL8 = models.BooleanField()
    winHL9 = models.BooleanField()
    winHL10 = models.BooleanField()
    resultHL1 = models.FloatField()
    resultHL2 = models.FloatField()
    resultHL3 = models.FloatField()
    resultHL4 = models.FloatField()
    resultHL5 = models.FloatField()
    resultHL6 = models.FloatField()
    resultHL7 = models.FloatField()
    resultHL8 = models.FloatField()
    resultHL9 = models.FloatField()
    resultHL10 = models.FloatField()
    choice1LF = make_booleanfield()
    choice2LF = make_booleanfield()
    choice3LF = make_booleanfield()
    choice4LF = make_booleanfield()
    choice5LF = make_booleanfield()
    choice6LF = make_booleanfield()
    choice7LF = make_booleanfield()
    choice8LF = make_booleanfield()
    choice9LF = make_booleanfield()
    choice10LF = make_booleanfield()
    winHL1LF = models.BooleanField()
    winHL2LF = models.BooleanField()
    winHL3LF = models.BooleanField()
    winHL4LF = models.BooleanField()
    winHL5LF = models.BooleanField()
    winHL6LF = models.BooleanField()
    winHL7LF = models.BooleanField()
    winHL8LF = models.BooleanField()
    winHL9LF = models.BooleanField()
    winHL10LF = models.BooleanField()
    resultHL1LF = models.FloatField()
    resultHL2LF = models.FloatField()
    resultHL3LF = models.FloatField()
    resultHL4LF = models.FloatField()
    resultHL5LF = models.FloatField()
    resultHL6LF = models.FloatField()
    resultHL7LF = models.FloatField()
    resultHL8LF = models.FloatField()
    resultHL9LF = models.FloatField()
    resultHL10LF = models.FloatField()
    real_group = models.IntegerField()
    ranking = models.IntegerField()
    real_rank = models.IntegerField()
    number_of_players = models.IntegerField()
    treatment_temp = models.StringField()
    treatment = models.StringField()

class Group(BaseGroup):
    pass


class Subsession(BaseSubsession):
    pass

# PAGES

class InformedConsentDE(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        player.language_picked = 'en'
        participant.language = player.language_picked
        return {
            'lang': participant.language
        }


class InformedConsentEN(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class InformedConsentWP(WaitPage):
    wait_for_all_groups = True
    title_text = "Please wait"
    body_text = "Waiting for other participants"

    @staticmethod
    def is_displayed(player: Player):
        return player.participant.language == 'en'


class InformedConsentWPde(WaitPage):
    wait_for_all_groups = True
    title_text = "Bitte warten"
    body_text = "Warten auf andere Teilnehmern"

    @staticmethod
    def is_displayed(player: Player):
        return player.participant.language == 'de'


class Instructions(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


# class RealEffort(Page):
#     pass
class Questionnaire(Page):
    form_model = "player"
    form_fields = ['questionnaire_age', 'questionnaire_gender', 'questionnaire_highestqualification',
                   'questionnaire_study', 'questionnaire_semester',
                   'questionnaire_smoker', 'questionnaire_health',
                   'questionnaire_exercisedays', 'questionnaire_exercisehours',
                   'questionnaire_insurancehealth', 'questionnaire_parentsplan',
                   'questionnaire_haftpflicht', 'questionnaire_carinsurance',
                   'questionnaire_disabilityinsurance', 'questionnaire_lifeinsurance',
                   'questionnaire_hausrat', 'questionnaire_legalinsurance',
                   'questionnaire_riskdriving', 'questionnaire_riskfinance',
                   'questionnaire_risksport', 'questionnaire_riskjob',
                   'questionnaire_riskhealth', 'questionnaire_risk',
                   'questionnaire_genderde',
                   'questionnaire_highestqualificationde',
                   'questionnaire_studyde', 'questionnaire_smokerde', 'questionnaire_healthde',
                   'questionnaire_insurancehealthde', 'questionnaire_parentsplande',
                   'questionnaire_haftpflichtde', 'questionnaire_carinsurancede',
                   'questionnaire_disabilityinsurancede', 'questionnaire_lifeinsurancede',
                   'questionnaire_hausratde', 'questionnaire_legalinsurancede',
                   'questionnaire_exercisedaysde', 'questionnaire_riskdrivingde', 'questionnaire_riskfinancede',
                   'questionnaire_risksportde', 'questionnaire_riskjobde', 'questionnaire_riskhealthde',
                   'questionnaire_riskde'
                   ]

    # @staticmethod
    # def before_next_page(self, timeout_happened):
    #     self.prolific_id = self.participant.label

    @staticmethod
    def get_form_fields(player: Player):
        if player.participant.language == 'de':
            return ['questionnaire_age', 'questionnaire_exercisedaysde',
                    'questionnaire_exercisehours',
                    'questionnaire_genderde',
                    'questionnaire_smokerde', 'questionnaire_healthde',
                    'questionnaire_insurancehealthde', 'questionnaire_lifeinsurancede',
                    'questionnaire_riskfinancede',
                    'questionnaire_risksportde', 'questionnaire_riskjobde', 'questionnaire_riskhealthde',
                    'questionnaire_riskde', 'questionnaire_riskdrivingde'

                    ]
        else:
            return ['questionnaire_age', 'questionnaire_gender',
                    'questionnaire_smoker', 'questionnaire_health',
                    'questionnaire_exercisedays', 'questionnaire_exercisehours',
                    'questionnaire_insurancehealth', 'questionnaire_lifeinsurance',
                    'questionnaire_riskdriving', 'questionnaire_riskfinance',
                    'questionnaire_risksport', 'questionnaire_riskjob',
                    'questionnaire_riskhealth', 'questionnaire_risk'
                    ]

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }

class RiskElicitation(Page):
    form_model = "player"
    form_fields = ['number_entered']

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class ResultsRisk(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        participant.number_entered_participant = player.number_entered
        rnum = random.random()
        if rnum <= C.probability_risk:
            player.win_risk = 1
            participant.win_risk_participant = player.win_risk
            player.result_risk = player.number_entered * \
                C.factor + C.initial - player.number_entered
            participant.result_risk_participant = player.result_risk
            return {
                'resultrisk': player.result_risk,
                'winrisk': player.win_risk,
                'numberentered': player.number_entered,
                'lang': participant.language
            }
        else:
            player.win_risk = 0
            participant.win_risk_participant = player.win_risk
            player.result_risk = C.initial - player.number_entered
            participant.result_risk_participant = player.result_risk
            return {
                'resultrisk': player.result_risk,
                'winrisk': player.win_risk,
                'numberentered': player.number_entered,
                'lang': participant.language
            }


class HL(Page):
    form_model = 'player'
    form_fields = ['choice%s' % i for i in range(1, 11)]

    @staticmethod
    def before_next_page(player, timeout_happened):
        wins = []
        results = []
        probs = []
        choices = [player.choice1, player.choice2, player.choice3, player.choice4, player.choice5,
                   player.choice6, player.choice7, player.choice8, player.choice9, player.choice10]
        for i in range(1, 11):
            rnum = random.random()
            probability = i / 10
            probs.append(probability)
            if rnum <= probability:
                win = 1
                wins.append(win)
            else:
                win = 0
                wins.append(win)
            choice = choices[i - 1]
            results.append(C.HLoutcomes[win][choice])
        player_var_list = [[probs[i], wins[i], results[i], choices[i]]
                           for i in range(len(probs))]
        pick_a_lottery = random.randint(0, 9)
        player.winHL = wins[pick_a_lottery]
        player.choiceHL = choices[pick_a_lottery]
        participant = player.participant
        participant.pick_a_lotto = pick_a_lottery + 1
        participant.choice_hl_part = player.choiceHL
        participant.win_hl_part = player.winHL
        participant.payoffHL_participant = player_var_list[pick_a_lottery][2]
        player.winHL1 = player_var_list[0][1]
        player.winHL2 = player_var_list[1][1]
        player.winHL3 = player_var_list[2][1]
        player.winHL4 = player_var_list[3][1]
        player.winHL5 = player_var_list[4][1]
        player.winHL6 = player_var_list[5][1]
        player.winHL7 = player_var_list[6][1]
        player.winHL8 = player_var_list[7][1]
        player.winHL9 = player_var_list[8][1]
        player.winHL10 = player_var_list[9][1]
        player.resultHL1 = player_var_list[0][2]
        player.resultHL2 = player_var_list[1][2]
        player.resultHL3 = player_var_list[2][2]
        player.resultHL4 = player_var_list[3][2]
        player.resultHL5 = player_var_list[4][2]
        player.resultHL6 = player_var_list[5][2]
        player.resultHL7 = player_var_list[6][2]
        player.resultHL8 = player_var_list[7][2]
        player.resultHL9 = player_var_list[8][2]
        player.resultHL10 = player_var_list[9][2]

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class HLLossFraming(Page):
    form_model = 'player'
    form_fields = ['choice%sLF' % i for i in range(1, 11)]

    @staticmethod
    def before_next_page(player, timeout_happened):
        wins = []
        results = []
        probs = []
        choices = [player.choice1LF, player.choice2LF, player.choice3LF, player.choice4LF, player.choice5LF,
                   player.choice6LF, player.choice7LF, player.choice8LF, player.choice9LF, player.choice10LF]
        for i in range(1, 11):
            rnum = random.random()
            probability = i / 10
            probs.append(probability)
            if rnum <= probability:
                win = 1
                wins.append(win)
            else:
                win = 0
                wins.append(win)
            choice = choices[i - 1]
            results.append(C.HLLFoutcomes[win][choice])
        player_var_list = [[probs[i], wins[i], results[i], choices[i]]
                           for i in range(len(probs))]
        pick_another_lottery = random.randint(0, 9)
        participant = player.participant
        player.choiceHLLF = choices[pick_another_lottery]
        player.winHLLF = wins[pick_another_lottery]
        participant.pick_another_lotto = pick_another_lottery + 1
        participant.choice_hllf_part = player.choiceHLLF
        participant.win_hllf_part = player.winHLLF
        participant.payoffHLLF_participant = C.HLLFinitial - \
            player_var_list[pick_another_lottery][2]
        player.winHL1LF = player_var_list[0][1]
        player.winHL2LF = player_var_list[1][1]
        player.winHL3LF = player_var_list[2][1]
        player.winHL4LF = player_var_list[3][1]
        player.winHL5LF = player_var_list[4][1]
        player.winHL6LF = player_var_list[5][1]
        player.winHL7LF = player_var_list[6][1]
        player.winHL8LF = player_var_list[7][1]
        player.winHL9LF = player_var_list[8][1]
        player.winHL10LF = player_var_list[9][1]
        player.resultHL1LF = player_var_list[0][2]
        player.resultHL2LF = player_var_list[1][2]
        player.resultHL3LF = player_var_list[2][2]
        player.resultHL4LF = player_var_list[3][2]
        player.resultHL5LF = player_var_list[4][2]
        player.resultHL6LF = player_var_list[5][2]
        player.resultHL7LF = player_var_list[6][2]
        player.resultHL8LF = player_var_list[7][2]
        player.resultHL9LF = player_var_list[8][2]
        player.resultHL10LF = player_var_list[9][2]

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class PartIWelcome(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class PartIIWelcome(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }

class AssignLosses(Page):
    @staticmethod
    def vars_for_template(player: Player):
        loss_amounts = [C.loss_1, C.loss_2, C.loss_3, C.loss_4, C.loss_5]
        random.shuffle(loss_amounts)
        participant = player.participant
        participant.loss_amount_1 = loss_amounts[0]
        participant.loss_amount_2 = loss_amounts[1]
        participant.loss_amount_3 = loss_amounts[2]
        participant.loss_amount_4 = loss_amounts[3]
        participant.loss_amount_5 = loss_amounts[4]
        return {
            'loss1': participant.loss_amount_1,
            'loss2': participant.loss_amount_2,
            'loss3': participant.loss_amount_3,
            'loss4': participant.loss_amount_4,
            'loss5': participant.loss_amount_5,
            'lang': participant.language
        }



page_sequence = [InformedConsentDE, Instructions, PartIWelcome, Questionnaire, RiskElicitation,
                 ResultsRisk, PartIIWelcome, HL, HLLossFraming, AssignLosses]  # , ]
