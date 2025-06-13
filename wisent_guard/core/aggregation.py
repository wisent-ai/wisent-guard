from enum import Enum

class ActivationAggregationMethod(Enum):
    LAST_TOKEN = 'last_token'
    MEAN = 'mean'
    MAX = 'max' 