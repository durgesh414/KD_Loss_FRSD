import sys
import logging
import collections
from copy import deepcopy


def merge(dict1, dict2):
    ''' Return a new dictionary by merging two dictionaries recursively. '''
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(value)
    return result

def fill_config(config):
    ''' Adjust the configuration by merging with a base configuration if present. '''
    base_cfg = config.pop('base', {})
    for sub, sub_cfg in config.items():
        if isinstance(sub_cfg, dict):
            config[sub] = merge(base_cfg, sub_cfg)
        elif isinstance(sub_cfg, list):
            config[sub] = [merge(base_cfg, c) for c in sub_cfg]
    return config


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0
        
    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
            print("Epochhh", self._epoch)
        return data

    def __len__(self):
        return len(self._dataloader)


class LoggerBuffer:
    def __init__(self, name, path, headers, screen_intvl=1):
        self.logger = self.get_logger(name, path)
        self.history = []
        self.headers = headers
        self.screen_intvl = screen_intvl

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y-%m-%d_%H-%M-%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)
        file_handler = logging.FileHandler(path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    def clean(self):
        self.history.clear()

    def update(self, msg):
        ''' Handle logging for a given message dictionary '''
        n = msg.pop('Iter')
        self.history.append(msg)
        novel_heads = [k for k in msg if k not in self.headers]
        if novel_heads:
            self.logger.warning('Items {} are not defined.'.format(novel_heads))
        missing_heads = [k for k in self.headers if k not in msg]
        if missing_heads:
            self.logger.warning('Items {} are missing.'.format(missing_heads))
        if n % self.screen_intvl == 0:
            screen_msg = self.construct_screen_message(n)
            self.logger.info(screen_msg)

    def construct_screen_message(self, n):
        screen_msg = ['Iter: {:5d}'.format(n)]
        for k, fmt in self.headers.items():
            vals = [msg[k] for msg in self.history[-self.screen_intvl:]
                    if k in msg]
            v = sum(vals) / len(vals)
            screen_msg.append(('{}: {'+fmt+'}').format(k, v))
        return ', '.join(screen_msg)

