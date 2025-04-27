from otree.api import *


author = 'Your name here'
doc = """
Your app description
"""


class Constants(BaseConstants):
    name_in_url = 'final_screen'
    players_per_group = None
    num_rounds = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# FUNCTIONS
# PAGES
class MyPage(Page):
    pass


page_sequence = [MyPage]
