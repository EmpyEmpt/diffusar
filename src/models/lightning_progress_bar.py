# Docs for tqdm-telegram: https://tqdm.github.io/docs/contrib.telegram/

from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.contrib.telegram import tqdm
import sys


class TelegramProgressBar(TQDMProgressBar):
    def __init__(self, chat_id, token):
        super().__init__()
        self.chat_id = chat_id
        self.token = token

    def init_train_tqdm(self):
        bar = tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            token=self.token,
            chat_id=self.chat_id
        )
        return bar
