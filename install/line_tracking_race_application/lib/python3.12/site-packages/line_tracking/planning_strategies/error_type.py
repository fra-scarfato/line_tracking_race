from enum import Enum


# Enum defining different error types the PID
#   can be asked to correct
class ErrorType(Enum):
    OFFSET = 0
    ANGLE = 1
