import datetime


class TimeElapsed:
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def get_time_elapsed(self):
        return datetime.datetime.now() - self.start_time

    def reset_time(self):
        self.start_time = datetime.datetime.now()

    def __str__(self):
        return str(self.get_time_elapsed())

    def __repr__(self):
        return str(self.get_time_elapsed())
