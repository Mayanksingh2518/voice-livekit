from datetime import date, timedelta

import pytest

from services import scheduler


@pytest.fixture(autouse=True)
def clean_bookings():
    scheduler.reset()
    yield
    scheduler.reset()


def _next_open_day(offset: int = 2) -> date:
    day = scheduler.now().date() + timedelta(days=offset)
    while not scheduler._is_open(day):
        day += timedelta(days=1)
    return day


class TestResolveDate:
    def test_iso(self):
        assert scheduler.resolve_date("2026-10-15") == date(2026, 10, 15)

    def test_relative_keywords(self):
        today = date(2026, 9, 12)
        assert scheduler.resolve_date("today", today=today) == today
        assert scheduler.resolve_date("tomorrow", today=today) == date(2026, 9, 13)
        assert scheduler.resolve_date("day after tomorrow", today=today) == date(2026, 9, 14)

    def test_weekday_resolves_forward(self):
        saturday = date(2026, 9, 12)
        assert scheduler.resolve_date("tuesday", today=saturday) == date(2026, 9, 15)

    def test_same_weekday_goes_to_next_week(self):
        saturday = date(2026, 9, 12)
        assert scheduler.resolve_date("saturday", today=saturday) == date(2026, 9, 19)

    def test_unparseable_raises(self):
        with pytest.raises(scheduler.SchedulingError):
            scheduler.resolve_date("sometime next quarter")

    def test_empty_raises(self):
        with pytest.raises(scheduler.SchedulingError):
            scheduler.resolve_date("")


class TestAvailability:
    def test_returns_requested_count(self):
        slots = scheduler.available_slots("endocrinology", on=_next_open_day(), limit=3)
        assert len(slots) == 3

    def test_is_deterministic(self):
        day = _next_open_day()
        first = scheduler.available_slots("endocrinology", on=day, limit=3)
        second = scheduler.available_slots("endocrinology", on=day, limit=3)
        assert [s.key for s in first] == [s.key for s in second]

    def test_window_filter_respected(self):
        slots = scheduler.available_slots("endocrinology", on=_next_open_day(), window="morning")
        assert all(s.start.hour < 13 for s in slots)

    def test_sunday_is_skipped(self):
        day = scheduler.now().date()
        while day.weekday() != 6:
            day += timedelta(days=1)
        slots = scheduler.available_slots("endocrinology", on=day, limit=1)
        assert slots and slots[0].start.date() != day


class TestBooking:
    def test_books_and_returns_confirmation(self):
        day = _next_open_day()
        booking = scheduler.book("P001", "Anita Sharma", "endocrinology", day.isoformat())
        assert booking.confirmation_id.startswith("APT-")
        assert booking.slot.start.date() == day

    def test_second_booking_takes_a_different_slot(self):
        day = _next_open_day()
        first = scheduler.book("P001", "A", "endocrinology", day.isoformat())
        second = scheduler.book("P002", "B", "endocrinology", day.isoformat())
        assert first.slot.key != second.slot.key

    def test_past_date_rejected(self):
        yesterday = scheduler.now().date() - timedelta(days=1)
        with pytest.raises(scheduler.SchedulingError, match="past"):
            scheduler.book("P001", "A", "endocrinology", yesterday.isoformat())

    def test_beyond_horizon_rejected(self):
        far = scheduler.now().date() + timedelta(days=scheduler.BOOKING_HORIZON_DAYS + 5)
        with pytest.raises(scheduler.SchedulingError, match="days ahead"):
            scheduler.book("P001", "A", "endocrinology", far.isoformat())

    def test_exhausted_day_offers_alternatives(self):
        day = _next_open_day()
        for _ in range(200):
            try:
                scheduler.book("PX", "X", "endocrinology", day.isoformat())
            except scheduler.SlotUnavailable as exc:
                assert exc.alternatives, "alternatives must be offered, not an empty failure"
                assert all(s.start.date() != day for s in exc.alternatives)
                return
        pytest.fail("expected the day to fill up")

    def test_unknown_specialty_falls_back_to_default(self):
        day = _next_open_day()
        booking = scheduler.book("P001", "A", "cardio-thoracic wizardry", day.isoformat())
        assert booking.slot.specialty == scheduler.DEFAULT_SPECIALTY

    def test_relative_date_accepted_from_model(self):
        """Models pass 'tuesday' rather than YYYY-MM-DD often enough that it must work."""
        target = _next_open_day()
        booking = scheduler.book(
            "P001", "A", "endocrinology", target.strftime("%A").lower()
        )
        assert booking.slot.start.date() == target

    def test_closed_day_says_so(self):
        day = scheduler.now().date()
        while day.weekday() != 6:
            day += timedelta(days=1)
        with pytest.raises(scheduler.SlotUnavailable, match="closed") as exc:
            scheduler.book("P001", "A", "endocrinology", day.isoformat())
        assert exc.value.alternatives, "a closed day must still offer alternatives"
