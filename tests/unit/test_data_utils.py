"""
Unit tests for data utilities.

parse_datetime feeds the behavioural analyser's time-of-day rules ("unusual
hour", "late night 2-5 AM"), so a timezone error here changes fraud decisions
rather than just formatting.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.utils.data_utils import parse_datetime


class TestParseDatetime:
    def test_epoch_seconds_are_interpreted_as_utc(self):
        """
        Epoch seconds are UTC by definition. Without an explicit tz they are
        read in the server's local zone, shifting every timestamp by the
        machine's offset - a bug that only shows up off UTC machines.
        """
        parsed = parse_datetime(1738571400)

        assert parsed == datetime(2025, 2, 3, 8, 30, tzinfo=UTC)
        assert parsed.tzinfo is not None

    def test_epoch_milliseconds_are_detected(self):
        assert parse_datetime(1738571400000) == parse_datetime(1738571400)

    def test_offset_is_honoured_not_discarded(self):
        """
        "+05:30" must shift the instant. The previous implementation stripped
        the offset with a regex before parsing, silently reading this as
        11:30 UTC instead of 06:00 UTC.
        """
        parsed = parse_datetime("2026-02-03T11:30:00+05:30")

        assert parsed == datetime(2026, 2, 3, 6, 0, tzinfo=UTC)

    def test_zulu_suffix_is_utc(self):
        assert parse_datetime("2026-02-03T11:30:00Z") == datetime(2026, 2, 3, 11, 30, tzinfo=UTC)

    def test_negative_offset_is_honoured(self):
        assert parse_datetime("2026-02-03T11:30:00-05:00") == datetime(
            2026, 2, 3, 16, 30, tzinfo=UTC
        )

    @pytest.mark.parametrize(
        "value",
        [
            "2026-02-03T11:30:00",
            "2026-02-03 11:30:00",
            "2026-02-03T11:30:00.123456",
        ],
    )
    def test_naive_strings_are_treated_as_utc(self, value):
        parsed = parse_datetime(value)

        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    def test_date_only(self):
        assert parse_datetime("2026-02-03") == datetime(2026, 2, 3, tzinfo=UTC)

    def test_legacy_slash_format_is_day_first(self):
        assert parse_datetime("03/02/2026") == datetime(2026, 2, 3, tzinfo=UTC)

    def test_aware_datetime_passes_through_unchanged(self):
        original = datetime(2026, 2, 3, 11, 30, tzinfo=timezone(timedelta(hours=2)))

        assert parse_datetime(original) is original

    def test_naive_datetime_is_made_aware(self):
        parsed = parse_datetime(datetime(2026, 2, 3, 11, 30))

        assert parsed == datetime(2026, 2, 3, 11, 30, tzinfo=UTC)

    @pytest.mark.parametrize("value", ["", "not a date", "2026-13-45"])
    def test_unparseable_returns_default(self, value):
        assert parse_datetime(value, default=None) is None

    def test_result_is_comparable_with_aware_now(self):
        """
        Every return value must be aware. A naive one raises TypeError the
        moment it is compared against the aware timestamps used elsewhere.
        """
        assert parse_datetime("2026-02-03T11:30:00") < datetime.now(UTC) + timedelta(days=365 * 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
