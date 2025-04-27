from otree.api import *


doc = """
Your app description
"""


class C(BaseConstants):
    NAME_IN_URL = 'welcomepage'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


# def gender_integerfield_en():
#     return models.IntegerField(
#         choices=[[1, 'Female'], [0, 'Male'], [
#             2, 'Diverse'], [-1, 'Prefer not to say']],
#         widget=widgets.RadioSelectHorizontal,
#         label="",
#     )


# def gender_integerfield_de():
#     return models.IntegerField(
#         choices=[[1, 'weiblich'], [0, 'männlich'],
#                  [2, 'divers'], [-1, 'keine Antwort']],
#         widget=widgets.RadioSelectHorizontal,
#         label="",
#     )


class Player(BasePlayer):
    language_picked = models.StringField()
    # gender_en = gender_integerfield_en()
    # gender_de = gender_integerfield_de()
    # gender = gender_integerfield_en()

# PAGES


class Welcome(Page):
    @staticmethod
    def vars_for_template(player: Player):
        player.language_picked = 'en'
        participant = player.participant
        participant.language = player.language_picked
        return {
            'lang': participant.language
        }
    # @staticmethod
    # def live_method(player, data):
    #     participant = player.participant
    #     if data['which_lang'] == 'german':
    #         player.language_picked = str(data['value'])
    #         participant.language = player.language_picked
    #     else:
    #         player.language_picked = str(data['value'])
    #         participant.language = player.language_picked


# class Testing(Page):
#     @staticmethod
#     def vars_for_template(player: Player):
#         participant = player.participant
#         return {
#             'lang': participant.language
#         }


#
# class Questionnaire(Page):
#     form_model = "player"
#     form_fields = ['gender_de', 'gender_en']
#
#     @staticmethod
#     def get_form_fields(player: Player):
#         if player.participant.language == 'de':
#             return ['gender_de']
#         else:
#             return ['gender_en']
#
#     @staticmethod
#     def vars_for_template(player: Player):
#         participant = player.participant
#         if player.participant.language == 'de':
#             player.gender_en = player.gender_de
#         return {
#             'lang': participant.language
#         }


page_sequence = [Welcome]
