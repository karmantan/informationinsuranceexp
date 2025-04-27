from otree.api import *


doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'realeffort'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1



class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass

def make_field():
    return models.StringField()

class Player(BasePlayer):
    for i in range(1,21):
        locals()['answer'+str(i)] = make_field()
    del i


# PAGES
class MyPage(Page):
    form_model = "player"
    form_fields = ['answer'+str(j) for j in range(1, 21)]

    @staticmethod
    def error_message(player: Player, values):
        solutions = dict(
            answer1='Selbstbeobachtung',
            answer2='Gegenmittel',
            answer3='zweideutig',
            answer4='Angeber',
            answer5='lindern',
            answer6='locken',
            answer7='verabscheuen',
            answer8='verschmelzen',
            answer9='verbinden',
            answer10='fliehen',
            answer11='freundlich',
            answer12='sterbend',
            answer13='Abweichung',
            answer14='Meineid',
            answer15='handhabbar',
            answer16='Gerinnung',
            answer17='glucksen',
            answer18='Menschenfeind',
            answer19='Geschicklichkeit',
            answer20='unpassend'
        )
        error_messages = dict()

        for field_name in solutions:
            if values[field_name] != solutions[field_name]:
                error_messages[field_name] = 'Wrong answer, please check your input.'

        return error_messages


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        player.payoff = 500
        return {
            "payoff": player.payoff,
        }


page_sequence = [MyPage, Results]
