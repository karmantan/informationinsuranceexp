from otree.api import *
import random
from settings import LANGUAGE_CODE
doc = """
Your app description
"""

if LANGUAGE_CODE == 'de':
    from .lexicon_en import Lexicon
else:
    from .lexicon_de import Lexicon

which_language = {'en': False, 'de': False, 'zh': False}  # noqa
which_language[LANGUAGE_CODE[:2]] = True


class C(BaseConstants):
    NAME_IN_URL = 'informationtreatment'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 5
    no_char = 6
    char = [[[1.0, 7.8731, 1.0, 8.0],
             [1.0, 2.0, 1.0, 4.0],
             [0.0, 4.0, 0.0, 6.0],
             [0.0, 5.5, 0.0, 2.0],
             [1.0, 7.0, 0.0, 0.0]],
            [[0.0, 10.0, 0.0, 9.0],
             [0.0, 1.0, 0.0, 7.0],
             [1.0, 9.0, 1.0, 5.0],
             [1.0, 3.0, 1.0, 3.15637],
             [1.0, 6.0, 0.0, 1.0]],
            [[1.0, 5.0, 0.0, 1.0],
             [0.0, 7.8809, 1.0, 7.0],
             [1.0, 8.0, 0.0, 5.0],
             [0.0, 5.0, 1.0, 10.0],
             [0.0, 6.0, 1.0, 3.0]],
            [[0.0, 3.0, 1.0, 2.0],
             [1.0, 4.0, 1.0, 1.0],
             [0.0, 3.2938, 1.0, 4.0],
             [1.0, 2.0, 0.0, 1.0],
             [1.0, 1.0, 0.0, 3.0]],
            [[1.0, 6.0, 0.0, 5.0],
             [0.0, 7.0, 1.0, 4.0],
             [0.0, 4.0, 1.0, 3.0],
             [1.0, 9.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 9.285]]]
    baseline_prob = 0.5
    beta = [0.7, 0.35, -0.6, -0.45]
    var1effect_round1 = 0.168187772*100
    var3effect_round1 = 0.145656306*100  # negative
    var2effect_0 = 0
    var2effect_2 = 0.168187772*100
    var2effect_4 = 0.302183889*100
    var2effect_6 = 0.390903179*100
    var2effect_8 = 0.442675824*100
    var2effect_10 = 0.470687769*100
    var4effect_0 = 0
    var4effect_2 = 0.210949503*100
    var4effect_4 = 0.358148935*100
    var4effect_6 = 0.437026644*100
    var4effect_8 = 0.473403006*100
    var4effect_10 = 0.489013057*100
    treatments = ['baseline', 'fullinfo', 'varinfo', 'neginfo', 'posinfo']
    template = 'characteristics.html'
    earningseachround = 100
    survey_reward = 50
    conversion = 100
    belief_reward = 50


class Player(BasePlayer):
    var1 = models.FloatField()
    var2 = models.FloatField()
    var3 = models.FloatField()
    var4 = models.FloatField()
    prob = models.FloatField()
    belief = models.FloatField()
    WTP = models.FloatField()
    insurance = models.FloatField()
    # number_entered = models.FloatField(min=0, max=1000, label="")
    result_risk = models.FloatField()
    win_risk = models.BooleanField()
    treatment_temp = models.StringField()
    treatment = models.StringField()
    ranking = models.FloatField()
    real_rank = models.FloatField()
    real_group = models.FloatField()
    timer_char1 = models.FloatField()
    timer_char2 = models.FloatField()
    timer_char3 = models.FloatField()
    timer_char4 = models.FloatField()
    timer_char7 = models.FloatField()
    timer_faq_coverage = models.FloatField()
    timer_faq_noinsurance = models.FloatField()
    timer_faq_howitworks = models.FloatField()
    timer_faq_choiceaffectsprice = models.FloatField()
    belief_correct = models.BooleanField()
    result_belief = models.FloatField()
    state = models.StringField()
    number_of_players = models.IntegerField()
    payoff_this_round = models.FloatField()
    random_number = models.FloatField()
    final_payoff = models.FloatField()
    purchased_coverage = models.BooleanField()
    loss_amount = models.FloatField()
    display_payoff_euro = models.FloatField()


class Group(BaseGroup):
    pass


class Subsession(BaseSubsession):
    pass


def creating_session(subsession: Subsession):
    no_of_players = len(subsession.get_players())
    number_of_groups = no_of_players / 5
    number_of_groups_plus_one = number_of_groups + 1
    treatments_multiple = C.treatments * int(number_of_groups_plus_one)

    def chunks(list, n):
        for i in range(0, len(list), n):
            yield list[i:i + n]
    sort_into_fives = list(chunks(treatments_multiple, 5))
    for a in sort_into_fives:
        random.shuffle(a)
    treatments_shuffled = []
    for li in sort_into_fives:
        for item in li:
            treatments_shuffled.append(item)
    if subsession.round_number == 1:
        count = 0
        for player in subsession.get_players():
            participant = player.participant
            participant.number_of_players = no_of_players
            player.treatment = treatments_shuffled[count]
            participant.treatment = player.treatment
            count += 1

    players_grouped_into_25 = list(chunks(subsession.get_players(), 25))

    for bpg in range(len(players_grouped_into_25)):
        big_player_group = players_grouped_into_25[bpg]
        players_grouped_into_fives = list(chunks(big_player_group, 5))
        for rn in range(C.NUM_ROUNDS):
            r = rn+1
            char_rn_mult = C.char[rn]*int(number_of_groups_plus_one)
            beta_rn_mult = C.beta*int(number_of_groups_plus_one)
            # if len(big_player_group) == 25:
            if bpg == 0:
                pl_group_count = 0
                for pl_group in players_grouped_into_fives:
                    for pl in pl_group:
                        pl_in_round = pl.in_round(r)
                        pl_in_round.var1 = char_rn_mult[pl_group_count][0]
                        pl_in_round.var2 = char_rn_mult[pl_group_count][1]
                        pl_in_round.var3 = char_rn_mult[pl_group_count][2]
                        pl_in_round.var4 = char_rn_mult[pl_group_count][3]
                        pl_in_round.prob = probability_generator(
                            multiply_vector(char_rn_mult[pl_group_count], beta_rn_mult))
                    pl_group_count += 1
            else:
                no_odd_groups = len(players_grouped_into_fives)
                random_integers = random.Random(
                    no_odd_groups).sample(range(0, 5), no_odd_groups)
                for p in range(len(players_grouped_into_fives)):
                    pl_group = players_grouped_into_fives[p]
                    for pl in pl_group:
                        random_integer = random.Random().sample(
                            range(0, 5), 1)[0]
                        pl_in_round = pl.in_round(r)
                        pl_in_round.var1 = char_rn_mult[random_integer][0]
                        pl_in_round.var2 = char_rn_mult[random_integer][1]
                        pl_in_round.var3 = char_rn_mult[random_integer][2]
                        pl_in_round.var4 = char_rn_mult[random_integer][3]
                        pl_in_round.prob = probability_generator(
                            multiply_vector(char_rn_mult[random_integer], beta_rn_mult))

                # no_odd_groups = len(players_grouped_into_fives)
                # random_integers = random.Random(no_odd_groups).sample(range(0, 5), no_odd_groups)
                # for p in range(len(players_grouped_into_fives)):
                #     pl_group = players_grouped_into_fives[p]
                #     for pl in pl_group:
                #         pl_in_round = pl.in_round(r)
                #         pl_in_round.var1 = char_rn_mult[random_integers[p]][0]
                #         pl_in_round.var2 = char_rn_mult[random_integers[p]][1]
                #         pl_in_round.var3 = char_rn_mult[random_integers[p]][2]
                #         pl_in_round.var4 = char_rn_mult[random_integers[p]][3]
                #         pl_in_round.prob = probability_generator(multiply_vector(char_rn_mult[random_integers[p]], beta_rn_mult))


def multiply_vector(x, y):
    sum = 0
    for i in range(0, len(x)):
        ans = x[i-1]*y[i-1]
        sum += ans
    return sum


def probability_generator(x):
    prob = 2.71828**x/(1+2.71828**x)
    return prob


class RiskInformation(Page):

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class Beliefs(Page):
    form_model = 'player'
    form_fields = ['belief']

    @staticmethod
    def live_method(player, data):
        if data['which_char'] == 'char_1':
            player.timer_char1 = int(data['value'])
        elif data['which_char'] == 'char_2':
            player.timer_char2 = int(data['value'])
        elif data['which_char'] == 'char_3':
            player.timer_char3 = int(data['value'])
        elif data['which_char'] == 'char_4':
            player.timer_char4 = int(data['value'])
        else:
            player.timer_char7 = int(data['value'])

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        player.treatment = participant.treatment
        player.number_of_players = participant.number_of_players
        number_of_risk_groups = player.number_of_players / 5
        number_of_risk_groups_plus_one = number_of_risk_groups + 1
        treatments_multiple = C.treatments * \
            int(number_of_risk_groups_plus_one)
        n = len(treatments_multiple)
        m = C.NUM_ROUNDS
        return {
            "var1": player.var1,
            "var2": player.var2,
            "var3": player.var3,
            "var4": player.var4,
            "prob": player.prob*100,
            'Lexicon': Lexicon,
            'lang': participant.language
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        if abs(player.belief - 100 * player.prob) <= 10:
            player.result_belief = C.belief_reward
            player.belief_correct = 1
        else:
            player.result_belief = 0
            player.belief_correct = 0


class WillingnesstoPay(Page):
    form_model = 'player'
    form_fields = ['WTP']

    @staticmethod
    def live_method(player, data):
        if data['which_char'] == 'char_coverage':
            player.timer_faq_coverage = int(data['value'])
        elif data['which_char'] == 'char_noinsurnace':
            player.timer_faq_noinsurance = int(data['value'])
        elif data['which_char'] == 'char_howitworks':
            player.timer_faq_howitworks = int(data['value'])
        else:
            player.timer_faq_choiceaffectsprice = int(data['value'])

    @staticmethod
    def before_next_page(player, timeout_happened):
        random_number_temp = random.randint(1, player.loss_amount)
        player.random_number = random_number_temp
        if player.random_number <= player.WTP:
            player.insurance = player.random_number
            player.purchased_coverage = 1
        else:
            player.insurance = player.random_number
            player.purchased_coverage = 0

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


class Insurance(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            "wtp": player.WTP,
            "insurance": player.insurance,
            "var1": player.var1,
            "var2": player.var2,
            "var3": player.var3,
            "var4": player.var4,
            "prob": player.prob,
            "lang": participant.language
        }


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        rnum = random.random()
        if player.purchased_coverage == 0:
            if rnum <= player.prob:
                player.state = "ill"
                player.payoff_this_round = C.earningseachround + \
                    player.result_belief - player.loss_amount
            else:
                player.state = "healthy"
                player.payoff_this_round = C.earningseachround + player.result_belief
        else:
            if rnum <= player.prob:
                player.state = "ill"
                player.payoff_this_round = C.earningseachround + \
                    player.result_belief - player.insurance
            else:
                player.state = "healthy"
                player.payoff_this_round = C.earningseachround + \
                    player.result_belief - player.insurance
        return {
            "insurance": player.purchased_coverage,
            "state": player.state,
            "var1": player.var1,
            "var2": player.var2,
            "var3": player.var3,
            "var4": player.var4,
            "prob": player.prob,
            "lang": participant.language
        }


class FinalResults(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        player_payoff_in_diff_rounds = []
        for player_in_diff_rounds in player.in_all_rounds():
            player_payoff_in_diff_rounds.append(
                player_in_diff_rounds.payoff_this_round)
        player.participant.part3payoff = sum(player_payoff_in_diff_rounds)
        beliefs_round = []
        wtp_round = []
        insurance_round = []
        state_round = []
        results_round = []
        beliefs_correct_round = []
        insurance_purchases = []
        potential_losses = []
        insurance_prices = []
        for j in range(C.NUM_ROUNDS):
            i = j+1
            player_round = player.in_round(i)
            beliefs_round.append(player_round.result_belief)
            wtp_round.append(player_round.WTP)
            insurance_round.append(player_round.insurance)
            state_round.append(player_round.state)
            results_round.append(player_round.payoff_this_round)
            beliefs_correct_round.append(player_round.belief_correct)
            insurance_purchases.append(player_round.purchased_coverage)
            potential_losses.append(player_round.loss_amount)
        player_summary = [[k+1, C.earningseachround, beliefs_round[k], wtp_round[k], insurance_round[k], state_round[k],
                           results_round[k], beliefs_correct_round[k], insurance_purchases[k], potential_losses[k]] for k in range(len(results_round))]
        player.final_payoff = player.participant.result_risk_participant + player.participant.payoffHL_participant + \
            player.participant.payoffHLLF_participant + player.participant.part3payoff
        player.participant.payoff = cu(
            player.final_payoff)
        player.display_payoff_euro = round(
            player.final_payoff/C.conversion, 2)
        return {
            "part3payoff": player.participant.part3payoff,
            "endowment": C.earningseachround,
            "the_round_1": player_summary[0][0],
            "the_round_2": player_summary[1][0],
            "the_round_3": player_summary[2][0],
            "the_round_4": player_summary[3][0],
            "the_round_5": player_summary[4][0],
            "belief_round_1": player_summary[0][2],
            "belief_round_2": player_summary[1][2],
            "belief_round_3": player_summary[2][2],
            "belief_round_4": player_summary[3][2],
            "belief_round_5": player_summary[4][2],
            "belief_right_1": player_summary[0][7],
            "belief_right_2": player_summary[1][7],
            "belief_right_3": player_summary[2][7],
            "belief_right_4": player_summary[3][7],
            "belief_right_5": player_summary[4][7],
            "wtp_round_1": player_summary[0][3],
            "wtp_round_2": player_summary[1][3],
            "wtp_round_3": player_summary[2][3],
            "wtp_round_4": player_summary[3][3],
            "wtp_round_5": player_summary[4][3],
            "insurance_round_1": player_summary[0][4],
            "insurance_round_2": player_summary[1][4],
            "insurance_round_3": player_summary[2][4],
            "insurance_round_4": player_summary[3][4],
            "insurance_round_5": player_summary[4][4],
            "insurance_buy_1": player_summary[0][8],
            "insurance_buy_2": player_summary[1][8],
            "insurance_buy_3": player_summary[2][8],
            "insurance_buy_4": player_summary[3][8],
            "insurance_buy_5": player_summary[4][8],
            "loss_1": player_summary[0][9],
            "loss_2": player_summary[1][9],
            "loss_3": player_summary[2][9],
            "loss_4": player_summary[3][9],
            "loss_5": player_summary[4][9],
            "state_round_1": player_summary[0][5],
            "state_round_2": player_summary[1][5],
            "state_round_3": player_summary[2][5],
            "state_round_4": player_summary[3][5],
            "state_round_5": player_summary[4][5],
            "result_round_1": player_summary[0][6],
            "result_round_2": player_summary[1][6],
            "result_round_3": player_summary[2][6],
            "result_round_4": player_summary[3][6],
            "result_round_5": player_summary[4][6],
            "picked_a_lottery": player.participant.pick_a_lotto,
            "picked_another_lottery": player.participant.pick_another_lotto,
            "choseHL": player.participant.choice_hl_part,
            "choseHLLF": player.participant.choice_hllf_part,
            "wonHL": player.participant.win_hl_part,
            "wonHLLF": player.participant.win_hllf_part,
            "HL_payout": player.participant.payoffHL_participant,
            "HLLF_payout": player.participant.payoffHLLF_participant,
            "part2payout": player.participant.payoffHL_participant + player.participant.payoffHLLF_participant,
            "invested": player.participant.number_entered_participant,
            "invest_win": player.participant.win_risk_participant,
            "part1payoff": player.participant.result_risk_participant,
            "lang": player.participant.language
        }


class PartIIIWelcome(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        participant = player.participant
        losses = [participant.loss_amount_1,
                  participant.loss_amount_2,
                  participant.loss_amount_3,
                  participant.loss_amount_4,
                  participant.loss_amount_5]
        for m in range(C.NUM_ROUNDS):
            n = m + 1
            player_in_round = player.in_round(n)
            player_in_round.loss_amount = losses[m]

    @staticmethod
    def vars_for_template(player: Player):
        participant = player.participant
        return {
            'lang': participant.language
        }


page_sequence = [PartIIIWelcome, RiskInformation, Beliefs,
                 WillingnesstoPay, Insurance, Results, FinalResults]
