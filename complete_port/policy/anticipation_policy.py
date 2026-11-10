"""Submission-shaped policy. The actor is the trained network; this file is the
gateway contract: features from the observation and the metadata only, state
cleared at the start of an episode. Shaping and the extra critic are not here.
"""
from __future__ import annotations

import numpy as np

from gateway_features import ZoneMemory, update_features


class AnticipationPolicy:
    def __init__(self, building_meta_data, action_spec):
        self.meta = building_meta_data
        self.spec = action_spec
        names = list(building_meta_data["observation_field_names"])
        self.zone_temp = list(building_meta_data["zone_temp_indices"])
        oats = [i for i, name in enumerate(names) if name.endswith("outside_air_temperature_sensor")]
        if not oats:
            raise RuntimeError("observation has no outside air temperature")
        self.oat = oats[0]
        self.state = ZoneMemory(len(self.zone_temp))
        self.last = None
        self.steps = 0

    def reset(self) -> None:
        self.state.reset()
        self.last = None
        self.steps = 0

    def get_action(self, obs):
        features, summary = update_features(
            obs, self.meta, self.state, self.zone_temp, self.oat)
        self.last = summary
        self.steps += 1
        # The trained actor replaces this. Zeros are a legal action and do not
        # copy a controller; the check is that the features update from the
        # observation alone.
        _ = features
        continuous = np.zeros(self.spec["continuous_action"].shape, np.float32)
        discrete = np.zeros(self.spec["discrete_action"].shape, np.int32)
        return {"continuous_action": continuous, "discrete_action": discrete}
