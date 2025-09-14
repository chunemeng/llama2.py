import time


def time_in_ms():
    """Return current time in milliseconds."""
    return int(time.time() * 1000)


class TimeDict:
    def __init__(self):
        self.time_dict = {}
        self.num_dict = {}
        self.disable = False

    def reset(self):
        self.time_dict = {}
        self.num_dict = {}

    def store_time(self, name, delta):
        disable = self.disable
        if disable == False:
            self.time_dict[name] = self.time_dict.get(name, 0) + delta
            self.num_dict[name] = self.num_dict.get(name, 0) + 1

    def print_out(self):
        for k, v in zip(self.time_dict.keys(), self.time_dict.values()):
            print(f"{k}: {v:.2f} ms ({self.num_dict[k]} calls) avg: {v / self.num_dict[k]:.2f} ms")


global_time_dict = TimeDict()


def global_timing_report():
    global_time_dict.print_out()


def store_time(name, delta):
    global_time_dict.store_time(name, delta)

def clear_timing():
    global_time_dict.reset()
