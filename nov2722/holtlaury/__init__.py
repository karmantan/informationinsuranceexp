from otree.api import *
import random

doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'holtlaury'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 10
    A_win = 2
    A_lose = 1.60
    B_win = 3.85
    B_lose = 0.10


class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    choice = models.BooleanField(
        choices=[
            [True, 'A'],
            [False, 'B'],
        ],
        widget=widgets.RadioSelect
    )
    win = models.BooleanField()
    probability = models.FloatField()
    result = models.FloatField()

# PAGES

class MyPage(Page):
    form_model = "player"
    form_fields = ['choice']

class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        rnum = random.random()
        roundnumber = player.round_number
        probability = roundnumber/10
        player.probability = roundnumber/10
        if rnum <= probability:
            player.win = 1
            if player.choice == True:
                player.result = C.A_win
                player.payoff = player.result * 1000
                return {
                    "result": player.result,
                    "win": player.win,
                    "probability": player.probability,
                    "payoff": player.payoff,
                }
            else:
                player.result = C.B_win
                player.payoff = player.result * 1000


        else:
            player.win = 0
            if player.choice == True:
                player.result = C.A_lose
                player.payoff = player.result * 1000
                return {
                    "result": player.result,
                    "win": player.win,
                    "probability": player.probability,
                    "payoff": player.payoff,
                }
            else:
                player.result = C.B_lose
                player.payoff = player.result * 1000
                return {
                    "result": player.result,
                    "win": player.win,
                    "probability": player.probability,
                    "payoff": player.payoff,
                }

page_sequence = [MyPage, Results]
