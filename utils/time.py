import time


class TimeCollector:
    def __init__(self):
        self.times = {}

    def collect_time(self, name, clear_history=True):
        return TimeCollectorContextManager(self, name, clear_history)

    def __getattr__(self, name):
        if name in self.times:
            return self.times[name]
        else:
            raise AttributeError(f"'TimeCollector' object has no attribute '{name}'")


class TimeCollectorContextManager:
    def __init__(self, time_collector, name, clear_history):
        self.time_collector = time_collector
        self.name = name
        self.clear_history = clear_history

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        if self.clear_history:
            self.time_collector.times[self.name] = 0.0
        self.time_collector.times[self.name] += elapsed_time


tc = TimeCollector()
