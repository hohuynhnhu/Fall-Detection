# Fix: Mobile app not receiving GPS from fall events

## Issues found & fixed

### 1. Syntax error in `backend_client.py`

**File:** `src/services/backend_client.py:52`

Line 52 had `return self._cache` and `class GpsProvider:` concatenated into one line without a newline, causing `SyntaxError` on import.

**Fix:** Separated into two lines + removed duplicate `GpsProvider` class definition (lines 55-86 were an identical copy).

### 2. Missing backend route `GET /events/falls/{id}`

**File:** `fall-detection-backend/app/api/events.py`

Mobile app calls `GET /events/falls/{id}` to fetch a single fall event by ID, but backend had no such route → returned `404 Not Found` → mobile could not retrieve `latitude`/`longitude`.

**Fix:** Added route `GET /events/falls/{event_id}` (lines 339-371) that returns `FallEventResponse` with `latitude` and `longitude`.

### 3. GPS not attached to FallEvent

**File:** `src/core/camera_worker.py:683-697`

The camera worker constructs `FallEvent` directly and posts via httpx — it bypassed `BackendClient.send_fall()` entirely. GPS was never set on the event.

**Fix:** Added `gps=GpsProvider.get().location()` to the `FallEvent` constructor + added `from services.backend_client import GpsProvider` import.

### 4. Debug logging added

**File:** `src/services/backend_client.py`

Added debug prints in `GpsProvider.location()` and `BackendClient.send_fall()` to verify GPS is being fetched and attached.

## GPS data flow

```
Desktop                          Backend                        Mobile
───────                          ───────                        ──────
GpsProvider.get().location()     POST /events/fall              GET /events/falls/{id}
  → geocoder.ip("me")              → store latitude, longitude    → parse latitude, longitude
  → GpsLocation(lat, lon)          → send FCM push with GPS       → display in UI
  → attach to FallEvent            → return in FallEventResponse
```

## Verification

After fixes, desktop output should show:
```
[GpsProvider] GPS fetched: [10.823, 106.6296]
```

And backend should respond to `GET /events/falls/{id}` with `latitude` and `longitude` fields.
