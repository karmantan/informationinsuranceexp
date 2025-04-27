from os import environ
#
SESSION_CONFIGS = [
    dict(
        name='theexperiment',
        display_name="NOV_22_EXP",
        # 'informationtreatment','final_screen'],
        app_sequence=['preexperiment',
                      'informationtreatment', 'final_screen'],
        num_demo_participants=50,
    ),
]


# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=0.01, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = ['number_entered_participant',
                      'result_risk_participant',
                      'win_risk_participant',
                      'payoffHL_participant',
                      'payoffHLLF_participant',
                      'part3payoff',
                      'pick_a_lotto',
                      'pick_another_lotto',
                      'choice_hl_part',
                      'choice_hllf_part',
                      'win_hl_part',
                      'win_hllf_part',
                      'loss_amount_1',
                      'loss_amount_2',
                      'loss_amount_3',
                      'loss_amount_4',
                      'loss_amount_5',
                      'language',
                      'treatment',
                      'number_of_players'
                      ]

SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
# LANGUAGE_CODE = environ.get('LANGUAGE_CODE', 'en')
LANGUAGE_CODE = "en"
#

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'GBP'
USE_POINTS = True
# USE_POINTS = False

ROOMS = [
    dict(
        name='prol',
        display_name='prolific_room',
    ),
]

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

# OTREE_REST_KEY = 'nov0522'

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '1946957904600'
