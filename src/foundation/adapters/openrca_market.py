"""OpenRCA Market adapter.

Market is a cloud marketplace system with 11 services per cloudbed.
Two cloudbeds (cloudbed-1, cloudbed-2) are available.
The data includes metric_service.csv and metric_container.csv.

Entity model:
- SERVICE entities from metric_service.csv
- CONTAINER entities from metric_container.csv
"""

import os
from typing import List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from .base import BaseAdapter
from .openrca_bank import OpenRCABankAdapter
from ..schema.entity import Entity, EntityType
from ..schema.relation import Relation, RelationType
from ..schema.event import ObservationEvent, Modality, EventBatch
from ..schema.episode import RootCauseLabel


class OpenRCAMarketAdapter(OpenRCABankAdapter):
    """Adapter for OpenRCA Market dataset.

    Market is similar to Bank but with cloud marketplace services.
    Two cloudbed instances available (cloudbed-1, cloudbed-2).

    Directory structure:
        Market/Market/{cloudbed}/
        ├── record.csv
        └── telemetry/
            └── {date}/
                ├── metric/
                │   ├── metric_service.csv
                │   └── metric_container.csv
                ├── log/
                └── trace/
    """

    system_name = "market"
    # Uses Bank adapter logic since Market has same data format.
    # Override methods here if Market data format diverges.
