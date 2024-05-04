import torch

from .utils import to_device


def default_post_processor(batches):
    return to_device(
        batches,
        torch.cuda.current_device(),
        non_blocking=True,
    )


class DataloaderPrefetcher:
    def __init__(self, dataloader=None, post_processor=default_post_processor):
        self.stream = torch.cuda.Stream(priority=-1)
        self.dataloader = dataloader
        self.post_processor = post_processor

    def __call__(self, dataloader=None, post_processor=default_post_processor):
        self.dataloader = dataloader
        self.post_processor = post_processor
        return self.__iter__()

    def preload(self):
        try:
            with torch.cuda.stream(self.stream):
                self.next_data = next(self.iterator)
                if self.post_processor:
                    self.next_data = self.post_processor(self.next_data)
        except StopIteration:
            self.next_data = None

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is None:
            raise StopIteration()
        data = self.next_data
        self.preload()
        return data
