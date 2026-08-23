from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional


@dataclass(frozen=True)
class SensorRange:
    link_name: str
    hand_part: str
    start: int
    end: int

    @property
    def sensor_ids(self) -> range:
        return range(self.start, self.end)


# This order mirrors tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER, but is kept
# local so the graph utilities can be imported without the heavier data stack.
XELA_LINK_SENSOR_COUNTS = (
    ("3aftc_palm_link", 30),
    ("link_15_4x4_palm_link", 16),
    ("link_14_4x4_palm_link", 16),
    ("0aftc_palm_link", 30),
    ("link_2_4x4_palm_link", 16),
    ("link_1A_4x4_palm_link", 16),
    ("link_1B_4x4_palm_link", 16),
    ("1aftc_palm_link", 30),
    ("link_6_4x4_palm_link", 16),
    ("link_5A_4x4_palm_link", 16),
    ("link_5B_4x4_palm_link", 16),
    ("2aftc_palm_link", 30),
    ("link_10_4x4_palm_link", 16),
    ("link_9A_4x4_palm_link", 16),
    ("link_9B_4x4_palm_link", 16),
    ("ahr_palm_2_4x6_palm_link", 24),
    ("ahr_palm_1_4x6_palm_link", 24),
    ("ahr_palm_3_4x6_palm_link", 24),
)

LINK_TO_HAND_PART: Dict[str, str] = {
    "3aftc_palm_link": "thumb",
    "link_15_4x4_palm_link": "thumb",
    "link_14_4x4_palm_link": "thumb",
    "0aftc_palm_link": "index_finger",
    "link_2_4x4_palm_link": "index_finger",
    "link_1A_4x4_palm_link": "index_finger",
    "link_1B_4x4_palm_link": "index_finger",
    "1aftc_palm_link": "middle_finger",
    "link_6_4x4_palm_link": "middle_finger",
    "link_5A_4x4_palm_link": "middle_finger",
    "link_5B_4x4_palm_link": "middle_finger",
    "2aftc_palm_link": "ring_finger",
    "link_10_4x4_palm_link": "ring_finger",
    "link_9A_4x4_palm_link": "ring_finger",
    "link_9B_4x4_palm_link": "ring_finger",
    "ahr_palm_2_4x6_palm_link": "palm",
    "ahr_palm_1_4x6_palm_link": "palm",
    "ahr_palm_3_4x6_palm_link": "palm",
}

HAND_PART_ORDER = (
    "thumb",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "palm",
)

HAND_PART_LABELS_RU: Dict[str, str] = {
    "thumb": "большой палец",
    "index_finger": "указательный палец",
    "middle_finger": "средний палец",
    "ring_finger": "безымянный палец",
    "palm": "ладонь",
}

PHYSICAL_BRIDGE_LINK_PAIRS = (
    ("3aftc_palm_link", "link_15_4x4_palm_link"),
    ("link_15_4x4_palm_link", "link_14_4x4_palm_link"),
    ("link_14_4x4_palm_link", "ahr_palm_1_4x6_palm_link"),
    ("0aftc_palm_link", "link_2_4x4_palm_link"),
    ("link_2_4x4_palm_link", "link_1A_4x4_palm_link"),
    ("link_1A_4x4_palm_link", "link_1B_4x4_palm_link"),
    ("link_1B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("1aftc_palm_link", "link_6_4x4_palm_link"),
    ("link_6_4x4_palm_link", "link_5A_4x4_palm_link"),
    ("link_5A_4x4_palm_link", "link_5B_4x4_palm_link"),
    ("link_5B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("2aftc_palm_link", "link_10_4x4_palm_link"),
    ("link_10_4x4_palm_link", "link_9A_4x4_palm_link"),
    ("link_9A_4x4_palm_link", "link_9B_4x4_palm_link"),
    # URDF coordinates place the ring-finger pad closest to palm-1.  Keeping
    # this bridge on palm-3 creates a visibly long, anatomically implausible
    # jump to the middle of the palm.
    ("link_9B_4x4_palm_link", "ahr_palm_1_4x6_palm_link"),
    ("ahr_palm_1_4x6_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("ahr_palm_2_4x6_palm_link", "ahr_palm_3_4x6_palm_link"),
)


def iter_sensor_ranges(
    link_sensor_counts: Iterable[tuple[str, int]] = XELA_LINK_SENSOR_COUNTS,
    link_to_part: Mapping[str, str] = LINK_TO_HAND_PART,
) -> Iterator[SensorRange]:
    start = 0
    for link_name, count in link_sensor_counts:
        if link_name not in link_to_part:
            raise KeyError(f"Missing hand part for Xela link {link_name!r}")
        end = start + count
        yield SensorRange(
            link_name=link_name,
            hand_part=link_to_part[link_name],
            start=start,
            end=end,
        )
        start = end


def _build_sensor_to_part() -> Dict[int, str]:
    sensor_to_part: Dict[int, str] = {}
    for sensor_range in iter_sensor_ranges():
        for sensor_id in sensor_range.sensor_ids:
            if sensor_id in sensor_to_part:
                raise ValueError(f"Sensor {sensor_id} is mapped more than once")
            sensor_to_part[sensor_id] = sensor_range.hand_part
    _validate_complete_sensor_mapping(sensor_to_part)
    return sensor_to_part


def _build_part_to_sensors(sensor_to_part: Mapping[int, str]) -> Dict[str, List[int]]:
    part_to_sensors: defaultdict[str, List[int]] = defaultdict(list)
    for sensor_id, hand_part in sorted(sensor_to_part.items()):
        part_to_sensors[hand_part].append(sensor_id)
    return dict(part_to_sensors)


def _validate_complete_sensor_mapping(sensor_to_part: Mapping[int, str], num_sensors: int = 368) -> None:
    expected_ids = set(range(num_sensors))
    actual_ids = set(sensor_to_part.keys())
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"Expected sensor ids 0..{num_sensors - 1}; missing={missing}, extra={extra}"
        )


def validate_against_xela_flatten_order(
    xela_flatten_order: Optional[Mapping[str, int]] = None,
) -> None:
    """Verify that the local graph order still matches the data preprocessing order.

    Passing `xela_flatten_order` keeps this function dependency-light. If omitted,
    it imports tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER lazily.
    """
    if xela_flatten_order is None:
        from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER

        xela_flatten_order = XELA_FLATTEN_ORDER

    local_order = dict(XELA_LINK_SENSOR_COUNTS)
    if dict(xela_flatten_order) != local_order:
        raise ValueError(
            "tactile_ssl.graph.utils.XELA_LINK_SENSOR_COUNTS does not match "
            "tactile_ssl.data.xela.utils.XELA_FLATTEN_ORDER"
        )
    _validate_complete_sensor_mapping(SENSOR_TO_HAND_PART)


SENSOR_TO_HAND_PART: Dict[int, str] = _build_sensor_to_part()
HAND_PART_TO_SENSORS: Dict[str, List[int]] = _build_part_to_sensors(SENSOR_TO_HAND_PART)
