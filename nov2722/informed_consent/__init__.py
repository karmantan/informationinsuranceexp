from otree.api import *


author = 'Your name here'
doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'my_informed_consent'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# FUNCTIONS
# PAGES
class MyPage_en(Page):
    pass


class FinalWaitPage(WaitPage):
    wait_for_all_groups = True


page_sequence = [MyPage_en, FinalWaitPage]
