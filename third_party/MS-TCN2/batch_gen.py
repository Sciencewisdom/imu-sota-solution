#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(
        self,
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        sample_rate,
        preload=False,
        chunk_len=0,
        fg_min_ratio=0.0,
        background_id=0,
        chunk_tries=30,
    ):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.preload = bool(preload)
        self.chunk_len = int(chunk_len)
        self.fg_min_ratio = float(fg_min_ratio)
        self.background_id = int(background_id)
        self.chunk_tries = int(chunk_tries)
        self._feat_cache = {}
        self._tgt_cache = {}
        self._rng = np.random.default_rng(0)

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        if self.preload:
            self._feat_cache = {}
            self._tgt_cache = {}
            for vid in self.list_of_examples:
                stem = vid.split('.')[0]
                feats = np.load(self.features_path + stem + '.npy')
                file_ptr = open(self.gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                classes = np.zeros(min(np.shape(feats)[1], len(content)), dtype=np.int64)
                for i in range(len(classes)):
                    classes[i] = int(self.actions_dict[content[i]])
                # Apply sample_rate here for speed.
                self._feat_cache[vid] = feats[:, :: self.sample_rate].astype(np.float32, copy=False)
                self._tgt_cache[vid] = classes[:: self.sample_rate]

    def _pick_chunk(self, feats: np.ndarray, tgt: np.ndarray):
        # feats: (D, T), tgt: (T,)
        T = int(tgt.shape[0])
        if self.chunk_len <= 0 or T <= self.chunk_len:
            return feats, tgt
        L = int(self.chunk_len)
        # Try to pick a chunk with enough foreground.
        for _ in range(max(1, self.chunk_tries)):
            a = int(self._rng.integers(0, T - L + 1))
            b = a + L
            y = tgt[a:b]
            fg = float(np.mean(y != self.background_id))
            if fg >= self.fg_min_ratio:
                return feats[:, a:b], y
        # Fallback to random chunk.
        a = int(self._rng.integers(0, T - L + 1))
        b = a + L
        return feats[:, a:b], tgt[a:b]

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            if self.preload:
                feats = self._feat_cache[vid]
                tgt = self._tgt_cache[vid]
            else:
                feats = np.load(self.features_path + vid.split('.')[0] + '.npy')[:, :: self.sample_rate]
                file_ptr = open(self.gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                tgt = np.zeros(min(np.shape(feats)[1], len(content)), dtype=np.int64)
                for i in range(len(tgt)):
                    tgt[i] = int(self.actions_dict[content[i]])
                tgt = tgt[:: self.sample_rate]
                feats = feats.astype(np.float32, copy=False)

            feats, tgt = self._pick_chunk(feats, tgt)
            batch_input.append(feats)
            batch_target.append(tgt)

        length_of_sequences = list(map(len, batch_target))
        max_len = int(max(length_of_sequences)) if length_of_sequences else 0
        # Use pinned memory to speed up H2D copies.
        batch_input_tensor = torch.zeros(
            len(batch_input), int(np.shape(batch_input[0])[0]), max_len, dtype=torch.float, pin_memory=True
        )
        batch_target_tensor = torch.ones(len(batch_input), max_len, dtype=torch.long, pin_memory=True) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max_len, dtype=torch.float, pin_memory=True)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
