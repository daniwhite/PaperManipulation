"""
Just defining a bunch of exceptions for use in exiting.
"""

class SimExit(Exception):
    pass

class SimTaskComplete(SimExit):
    pass

class SimFailure(SimExit):
    pass

class ContactBroken(SimFailure):
    pass

class SimStalled(SimFailure):
    pass

class SimTimedOut(SimFailure):
    pass
