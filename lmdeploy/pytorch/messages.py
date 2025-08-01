# Copyright (c) OpenMMLab. All rights reserved.
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from torch import Tensor

from lmdeploy.messages import GenerationConfig, LogitsProcessor
from lmdeploy.pytorch.disagg.request import MigrationRequest
from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs
from lmdeploy.utils import get_logger

from .block import LogicalTokenBlocks

logger = get_logger('lmdeploy')

# vlm input type from pipeline
InputEmbeddingType = List[np.ndarray]
InputEmbeddingRangeType = List[List[int]]


@dataclass
class InputEmbeddings:
    """InputEmbeddings."""
    embeddings: np.ndarray
    start: int
    end: int

    def move_position(self, offset: int = 0):
        if offset != 0:
            self.start += offset
            self.end += offset
        return self


@dataclass
class SamplingParam:
    """Sampling parameter."""
    top_p: float = 1.0
    top_k: int = 1
    min_p: float = 0.0
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: List[int] = field(default_factory=list)
    bad_words: List[int] = field(default_factory=list)
    max_new_tokens: int = 512
    min_new_tokens: int = 0
    response_format: Optional[str] = None
    logits_processors: Optional[List[LogitsProcessor]] = None
    out_logits: bool = False
    out_last_hidden_states: bool = False

    @classmethod
    def from_gen_config(self, gen_config: GenerationConfig):
        """From gen config."""
        min_new_tokens = gen_config.min_new_tokens or 0

        stop_words = gen_config.stop_token_ids or []
        bad_words = gen_config.bad_token_ids or []
        if gen_config.ignore_eos:
            bad_words += stop_words
            stop_words = []

        top_k = gen_config.top_k
        top_p = gen_config.top_p
        min_p = gen_config.min_p
        temperature = gen_config.temperature
        repetition_penalty = gen_config.repetition_penalty
        max_new_tokens = gen_config.max_new_tokens
        response_format = gen_config.response_format

        output_logits = gen_config.output_logits
        if output_logits:
            if (output_logits != 'all' or gen_config.max_new_tokens > 0):
                output_logits = None
                logger.warning('Pytorch Engine only support output_logits="all"'
                               ' with max_new_tokens=0')
        if gen_config.output_last_hidden_state is not None:
            logger.warning('Pytorch Engine does not support output last hidden states.')
        if top_p < 0 or top_p > 1.0:
            logger.warning('`top_p` has to be a float > 0 and < 1'
                           f' but is {top_p}')
            top_p = 1.0
        if min_p < 0 or min_p > 1.0:
            logger.warning('`min_p` has to be a float > 0 and < 1'
                           f' but is {min_p}')
            min_p = 0.0
        if temperature == 0:
            logger.warning('`temperature` is 0, set top_k=1.')
            temperature = 1.0
            top_k = 1
        if temperature < 0:
            logger.warning('`temperature` has to be a strictly'
                           f' positive value, but is {temperature}')
            temperature = 1.0
        if repetition_penalty <= 0:
            logger.warning('`repetition_penalty` has to be a strictly'
                           f' positive value, but is {repetition_penalty}')
            repetition_penalty = 1.0
        if max_new_tokens < 0:
            logger.warning('`max_new_tokens` has to be a strictly'
                           f' positive value, but is {max_new_tokens}')
            max_new_tokens = 512
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            logger.warning('`min_new_tokens` has to be '
                           'a int >=0 and <= `max_new_tokens`,'
                           f' but is {min_new_tokens}')
            min_new_tokens = 0
        return SamplingParam(top_p=top_p,
                             top_k=top_k,
                             min_p=min_p,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             ignore_eos=gen_config.ignore_eos,
                             random_seed=gen_config.random_seed,
                             stop_words=stop_words,
                             bad_words=bad_words,
                             response_format=response_format,
                             max_new_tokens=max_new_tokens,
                             min_new_tokens=min_new_tokens,
                             logits_processors=gen_config.logits_processors,
                             out_logits=(output_logits is not None))


class MessageStatus(enum.Enum):
    """Status of a sequence."""

    WAITING = enum.auto()
    RUNNING = enum.auto()
    STOPPED = enum.auto()
    ENDED = enum.auto()
    ABORTED = enum.auto()
    LOCKED = enum.auto()

    # PD Disaggregation
    # WAITING_MIGRATION: state of Unmigrated Requests
    # in both prefill and decode engines are tagged by
    # RUNNING_MIGRATION: state of Migrating Requests
    # in decode engine
    TO_BE_MIGRATED = enum.auto()
    WAITING_MIGRATION = enum.auto()
    RUNNING_MIGRATION = enum.auto()
    MIGRATION_LOCKED = enum.auto()
    MIGRATION_DONE = enum.auto()


_SEQ_COUNT = 0


def _new_msg_id():
    """Get a new message id."""
    global _SEQ_COUNT
    seq_id = _SEQ_COUNT
    _SEQ_COUNT += 1
    return seq_id


SeqMap = Dict[int, 'SchedulerSequence']


class SequenceManager:
    """Sequence manager."""

    def __init__(self) -> None:
        self._seq_map: SeqMap = dict()
        self._status_seq_map: Dict[MessageStatus, SeqMap] = dict()
        for status in MessageStatus:
            self._status_seq_map[status] = dict()

    def get_all_sequences(self):
        """Get all sequences."""
        return self._seq_map.values()

    def get_sequences(self, states: MessageStatus):
        """Get sequences."""
        return self._status_seq_map[states]

    def num_sequences(self, status: MessageStatus):
        """Num sequences."""
        return len(self.get_sequences(status))

    def add_sequence(self, seq: 'SchedulerSequence'):
        """Add sequence."""
        seq_id = seq.seq_id
        status = seq.status
        status_map = self._status_seq_map[status]
        self._seq_map[seq_id] = seq
        status_map[seq_id] = seq

    def remove_sequence(self, seq: 'SchedulerSequence'):
        """Remove sequence."""
        seq_id = seq.seq_id
        status = seq.status
        status_map = self._status_seq_map[status]
        self._seq_map.pop(seq_id)
        status_map.pop(seq_id)

    def update_sequence_status(self, seq: 'SchedulerSequence', new_status: MessageStatus):
        """Update status."""
        old_status = seq.status
        if new_status == old_status:
            return
        seq_id = seq.seq_id
        old_status_map = self._status_seq_map[old_status]
        new_status_map = self._status_seq_map[new_status]
        # may be remove by async_end
        if seq_id in old_status_map:
            old_status_map.pop(seq_id)
            new_status_map[seq_id] = seq


class SchedulerSession:
    """Scheduler session."""

    def __init__(self, session_id: int, block_size: int, seq_manager: SequenceManager = None) -> None:
        self.session_id = session_id
        self.block_size = block_size
        self.status: MessageStatus = MessageStatus.RUNNING
        self.sequences: SeqMap = dict()
        self.seq_manager = seq_manager

    def add_sequence(self,
                     token_ids: Tensor,
                     sampling_param: SamplingParam = None,
                     adapter_name: str = None,
                     return_logits: bool = False,
                     multimodals: MultiModalInputs = None,
                     input_embeddings: List[InputEmbeddings] = None,
                     migration_request: Optional[MigrationRequest] = None,
                     resp_cache: bool = False,
                     preserve_cache: bool = False) -> 'SchedulerSequence':
        """Add a new message."""
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.numpy()
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)
        if token_ids.ndim == 0:
            token_ids = token_ids.unsqueeze(0)
        if sampling_param is None:
            sampling_param = SamplingParam()

        seq = SchedulerSequence(
            seq_id=_new_msg_id(),
            session=self,
            history_cache=HistoryTokenIds(token_ids),
            num_new_tokens=0,
            sampling_param=sampling_param,
            adapter_name=adapter_name,
            arrive_time=time.time(),
            history_embeddings=HistoryEmbeddings(input_embeddings),
            history_multimodals=HistoryMultiModals(multimodals),
            return_logits=return_logits,
            migration_request=migration_request,
            resp_cache=resp_cache,
            preserve_cache=preserve_cache,
        )
        self.sequences[seq.seq_id] = seq
        if self.seq_manager is not None:
            self.seq_manager.add_sequence(seq)
        return seq

    def remove_sequence(self, seq: 'SchedulerSequence'):
        """Remove sequence."""
        assert seq.seq_id in self.sequences
        self.sequences.pop(seq.seq_id)
        if self.seq_manager is not None:
            self.seq_manager.remove_sequence(seq)


def _div_up(x, n):
    """Perform div up."""
    return (x + n - 1) // n


def _round_up(x, n):
    """Perform round up."""
    return _div_up(x, n) * n


class HistoryEmbeddings:
    """History embeddings."""

    def __init__(self, embeddings: List[InputEmbeddings] = None):
        self._embeddings: List[InputEmbeddings] = []
        if embeddings is not None:
            self._embeddings.extend(embeddings)

    def append(self, embeddings: List[InputEmbeddings]):
        self._embeddings.extend(embeddings)

    def clone(self):
        ret = HistoryEmbeddings(self._embeddings)
        return ret

    def copy(self):
        return self.clone()

    def get_step(self, step: int) -> int:
        """Get step before a whole image."""
        real_step = step
        num_all_images = len(self._embeddings)
        history_image_num = 0
        if num_all_images > 0:
            history_image_num = sum([1 for emb in self._embeddings if emb.end <= step])
            if history_image_num < num_all_images:
                emb = self._embeddings[history_image_num]
                # for case step in middle of an image
                if emb.start < step:
                    real_step = emb.start
        num_images = num_all_images - history_image_num
        return real_step, history_image_num, num_images

    @property
    def embeddings(self):
        """embeddings."""
        return self._embeddings

    def __len__(self):
        """Get num images."""
        return len(self._embeddings)

    def __getitem__(self, *args, **kwargs):
        """Get values."""
        return self._embeddings.__getitem__(*args, **kwargs)


class HistoryTokenIds:
    """History token ids."""
    ALLOC_SIZE = 512

    def __init__(self, token_ids: np.ndarray = None):
        if token_ids is None:
            self._token_ids = np.empty((self.ALLOC_SIZE, ), dtype=np.int64)
            self._num_real = 0
        else:
            self._token_ids = token_ids
            self._num_real = len(token_ids)

    def reserve(self, size: int):
        """Reserve cache."""
        num_tokens = len(self._token_ids)
        if num_tokens >= size:
            return
        reserve_size = _round_up(size - num_tokens, self.ALLOC_SIZE)
        new_token_ids = np.pad(self._token_ids, (0, reserve_size))
        self._token_ids = new_token_ids

    def get_real(self):
        """Get logical blocks."""
        return self._token_ids[:self._num_real]

    def __setitem__(self, *args, **kwargs):
        """Set values."""
        return self.get_real().__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        """Get values."""
        return self.get_real().__getitem__(*args, **kwargs)

    def append(self, token_ids: np.ndarray):
        """Append token ids."""
        num_tokens = len(token_ids)
        self.reserve(num_tokens + self._num_real)
        slice_start = self._num_real
        slice_end = slice_start + num_tokens
        self._num_real += num_tokens
        self._token_ids[slice_start:slice_end] = token_ids

    def __len__(self):
        """Get length."""
        return self._num_real

    def clone(self):
        """clone."""
        ret = HistoryTokenIds()
        ret.append(self.get_real())
        return ret

    def copy(self):
        """copy."""
        return self.clone()


class HistoryMultiModals:

    def __init__(self, multimodals: MultiModalInputs):
        if multimodals is None:
            multimodals = dict()
        self.multimodals = multimodals

    def get_datas(self, start=0, end=-1):
        """Get multimodals from prompts position [start, end)."""
        outs = dict()
        test_range = range(start, end)
        for modal_type, modal_datas in self.multimodals.items():
            data = []
            for modal_data in modal_datas:
                if (modal_data.start not in test_range and modal_data.end not in test_range):
                    continue
                data.append(modal_data)
            if len(data) > 0:
                outs[modal_type] = data
        return outs

    def add_inputs(self, input_mms: MultiModalInputs):
        """Add new inputs."""
        for modal_type, vals in input_mms.items():
            if modal_type in self.multimodals:
                self.multimodals[modal_type] += vals
            else:
                self.multimodals[modal_type] = vals

    def empty(self):
        if len(self.multimodals) == 0:
            return 0

        return all(len(vals) == 0 for vals in self.multimodals)

    @staticmethod
    def update_multimodals(input_mms: MultiModalInputs, prev_len: int):
        """Update multimodals."""
        for vals in input_mms.values():
            for val in vals:
                val.start += prev_len
                val.end += prev_len
        return input_mms

    def get_encoder_len(self, start=0, end=-1):
        """Get lens of encoder."""
        test_range = range(start, end)
        out_len = 0
        for _, modal_datas in self.multimodals.items():
            for modal_data in modal_datas:
                if modal_data.encoder_len is None:
                    continue
                if (modal_data.start not in test_range and modal_data.end not in test_range):
                    continue
                out_len += modal_data.encoder_len
        return out_len


@dataclass
class SchedulerSequence:
    """Scheduler message."""
    seq_id: int
    session: SchedulerSession
    history_cache: HistoryTokenIds = field(default_factory=HistoryTokenIds)
    history_embeddings: HistoryEmbeddings = field(default_factory=HistoryEmbeddings)
    history_multimodals: HistoryMultiModals = field(default_factory=HistoryMultiModals)
    num_new_tokens: int = 0
    sampling_param: SamplingParam = field(default_factory=SamplingParam)
    logical_blocks: LogicalTokenBlocks = field(default_factory=LogicalTokenBlocks)
    adapter_name: str = None
    arrive_time: float = 0.0
    meta: Any = None
    return_logits: bool = False
    random_offsets: int = 0
    _status: MessageStatus = field(default=MessageStatus.WAITING, init=False)
    num_ignored_history: int = 0
    model_meta: Dict[str, Any] = None

    # For Disaggregation
    migration_request: Optional[MigrationRequest] = None
    resp_cache: bool = False
    preserve_cache: bool = False

    def __post_init__(self):
        """Post init."""
        self._num_history_ids: int = 0
        self._num_history_images: int = 0
        self._num_images: int = len(self.history_embeddings)
        self._num_token_ids: int = len(self.history_cache)

        self._num_history_cross: int = 0
        self._num_cross: int = self.history_multimodals.get_encoder_len(0, self._num_token_ids)

    @property
    def block_size(self) -> int:
        """Block size."""
        return self.session.block_size

    @property
    def history_len(self) -> int:
        """Get history length."""
        return self._num_history_ids

    @property
    def history_image_num(self) -> int:
        """Get history image number."""
        return self._num_history_images

    @property
    def history_image_token_len(self) -> int:
        """Get history image token length."""
        return sum([emb.end - emb.start for emb in self.history_embeddings[:self._num_history_images]])

    @property
    def session_id(self) -> int:
        """Get session id."""
        return self.session.session_id

    @property
    def token_ids(self) -> np.ndarray:
        """Token ids."""
        start = self.history_len
        end = start + self._num_token_ids
        return self.history_cache[start:end]

    @property
    def input_embeddings(self) -> List[InputEmbeddings]:
        """Get current embeddings."""
        start = self.history_image_num
        end = start + self._num_images
        return self.history_embeddings[start:end]

    @property
    def history_ids(self) -> np.ndarray:
        """History ids."""
        return self.history_cache[:self.history_len]

    @property
    def all_ids(self) -> np.ndarray:
        """Full token ids."""
        return self.history_cache[:self.num_all_ids]

    @property
    def num_history_ids(self):
        """Num history ids."""
        return self._num_history_ids

    @property
    def num_token_ids(self):
        return self._num_token_ids

    @property
    def num_images(self):
        return self._num_images

    @property
    def num_all_ids(self):
        """Num all tokens."""
        return self.history_len + self._num_token_ids

    @property
    def num_cross(self):
        """Num cross."""
        return self._num_cross

    @property
    def num_history_cross(self):
        """Num history cross."""
        return self._num_history_cross

    @property
    def num_blocks(self):
        """Num blocks."""
        return len(self.logical_blocks)

    @property
    def seq_manager(self) -> SequenceManager:
        """Sequence manager."""
        return self.session.seq_manager

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value: MessageStatus):
        self.seq_manager.update_sequence_status(self, value)
        self._status = value

    def num_all_tokens(self):
        """Num all tokens."""
        return self.num_all_ids

    def num_all_cross_tokens(self):
        """Num of all cross tokens."""
        return self._num_cross + self._num_history_cross

    def get_input_multimodals(self):
        """Get input multimodals."""
        start = self.num_history_ids
        end = self.num_all_ids
        return self.history_multimodals.get_datas(start, end)

    def update_token_ids(self,
                         token_ids: Tensor,
                         multimodals: MultiModalInputs = None,
                         embeddings: List[InputEmbeddings] = None,
                         model_meta: Dict[str, Any] = None,
                         append_tokens: bool = False):
        """Update token ids, old token ids will be added to history."""
        old_num_history_ids = self._num_history_ids

        # update history
        if not append_tokens:
            self._num_history_ids += self._num_token_ids

        # update history image nums
        self._num_history_images += self._num_images
        self._num_images = 0
        if embeddings is not None:
            new_embeddings = [emb.move_position(self._num_history_ids) for emb in embeddings]
            self._num_images = len(new_embeddings)
            self.history_embeddings.append(new_embeddings)

        # update multimodals
        if multimodals is not None:
            multimodals = HistoryMultiModals.update_multimodals(multimodals, self.num_all_ids)
            self.history_multimodals.add_inputs(multimodals)

        # cross
        self._num_history_cross += self._num_cross
        if multimodals is not None:
            self._num_cross = self.history_multimodals.get_encoder_len(old_num_history_ids, self._num_history_ids)
        else:
            self._num_cross = 0

        if model_meta is not None:
            self.model_meta = model_meta

        if isinstance(token_ids, Tensor):
            token_ids = token_ids.numpy()
        elif not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)
        if token_ids.ndim == 0:
            token_ids = token_ids[None]
        if append_tokens:
            self._num_token_ids += len(token_ids)
        else:
            self._num_token_ids = len(token_ids)
        self.history_cache.append(token_ids)
        self.random_offsets += 1
        self.arrive_time = time.time()

    def set_step(self, step: int):
        """Set step."""
        num_all_ids = self.num_all_ids
        # update step for vlm
        if len(self.history_embeddings) > 0:
            new_step, self._num_history_images, self._num_images = \
                self.history_embeddings.get_step(step)
            assert 0 <= new_step <= step
            step = new_step
        self._num_history_ids = step
        self._num_token_ids = num_all_ids - step
        self.num_ignored_history = min(step, self.num_ignored_history)

        self.model_meta = None

        # cross
        if self.history_multimodals is not None:
            self._num_history_cross = self.history_multimodals.get_encoder_len(0, self.num_history_ids)
            self._num_cross = self.history_multimodals.get_encoder_len(self._num_history_ids, num_all_ids)
