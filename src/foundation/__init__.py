"""RCAWorld-Foundation: Query-conditioned diagnostic world model.

Phase A implementation:
  - Universal Entity/Relation/ObservationEvent schema
  - OpenRCA Bank/Market/Telecom adapters
  - Heterogeneous entity tokenizer & metric encoder
  - Hierarchical RSSM with type-aware dynamics
  - Learned Onset Head replacing oracle window selection
  - Parallel component head: p(c|t) + p(c) dual paths
  - Flamingo-style gated cross-attention for LLM bridging
  - Comprehensive training losses and evaluation metrics
"""

from . import schema
from . import adapters
from . import models
from . import training
from . import evaluation
