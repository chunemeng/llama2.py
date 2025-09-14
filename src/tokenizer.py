import string
import struct
from typing import List

import torch

from ops.time_util import time_in_ms, store_time
from transformers import AutoTokenizer, Qwen3Model


class TokenIndex:
    def __init__(self, s: str, idx: int):
        self.str = s
        self.id = idx


class Tokenizer:
    def __init__(self):
        self.vocab = []  # List[str]
        self.vocab_scores = None  # torch.FloatTensor
        self.sorted_vocab = None  # List[TokenIndex]
        self.vocab_size = 0
        self.max_token_length = 0
        self.byte_pieces = [bytes([i]) for i in range(256)]

    def build_tokenizer(self, tokenizer_path: str, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab_scores = torch.zeros(vocab_size, dtype=torch.float32)
        self.vocab = [''] * vocab_size
        self.sorted_vocab = None

        with open(tokenizer_path, 'rb') as f:
            self.max_token_length = struct.unpack('i', f.read(4))[0]
            for i in range(vocab_size):
                self.vocab_scores[i] = struct.unpack('f', f.read(4))[0]
                length = struct.unpack('i', f.read(4))[0]
                self.vocab[i] = f.read(length).decode('utf-8')

    def decode(self, token_list: List[int]) -> str:
        t = time_in_ms()
        pieces = ''
        for token in token_list:
            piece = self.vocab[token]

            # handle raw byte tokens like '<0x01>'
            if piece.startswith('<0x') and piece.endswith('>'):
                byte_val = int(piece[3:-1], 16)
                piece = self.byte_pieces[byte_val].decode('latin1')
            pieces += piece

        t2 = time_in_ms()
        store_time('token_decode', t2 - t)
        return pieces

    def safe_print(self, piece: str):
        if piece is None or piece == '':
            return
        if len(piece) == 1:
            if piece not in string.printable:
                return
        print(piece, end='')

    def _build_sorted_vocab(self):
        if self.sorted_vocab is None:
            self.sorted_vocab = [TokenIndex(s, idx) for idx, s in enumerate(self.vocab)]
            self.sorted_vocab.sort(key=lambda x: x.str)

    def str_lookup(self, s: str) -> int:
        self._build_sorted_vocab()
        # binary search
        lo, hi = 0, self.vocab_size - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cmp = (self.sorted_vocab[mid].str > s) - (self.sorted_vocab[mid].str < s)
            if cmp == 0:
                return self.sorted_vocab[mid].id
            elif cmp < 0:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def encode(self, text: str, bos: bool = True, eos: bool = True):
        if text is None:
            raise ValueError("Cannot encode None text")

        self._build_sorted_vocab()
        tokens = []

        # add BOS token
        if bos:
            tokens.append(1)

        # dummy prefix token
        if text:
            dummy_prefix = self.str_lookup(" ")
            if dummy_prefix != -1:
                tokens.append(dummy_prefix)

        # UTF-8 byte processing
        str_buffer = bytearray()
        i = 0
        while i < len(text):
            c = text[i]
            b = c.encode('utf-8')
            str_buffer.extend(b)

            # try to match the whole buffer
            s = str_buffer.decode('utf-8', errors='ignore')
            idx = self.str_lookup(s)
            if idx != -1:
                tokens.append(idx)
                str_buffer.clear()
            else:
                # byte fallback for each UTF-8 byte
                for byte in b:
                    tokens.append(byte + 3)
                str_buffer.clear()
            i += 1

        # greedy merge according to vocab_scores
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                s = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                idx = self.str_lookup(s)
                if idx != -1 and self.vocab_scores[idx] > best_score:
                    best_score = self.vocab_scores[idx].item()
                    best_id = idx
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            tokens.pop(best_idx + 1)

        # add EOS token
        if eos:
            tokens.append(2)

        return tokens


class ModelTokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.vocab_size = self.tokenizer.vocab_size
        self.bos_id = 151644
        self.eos_id = 151645
        self.pad_id = self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = True):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)