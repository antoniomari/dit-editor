from typing import List, Optional, TypedDict, Union

import torch


class QKVCache(TypedDict):
    query: List[torch.Tensor]
    key: List[torch.Tensor]
    value: List[torch.Tensor]
