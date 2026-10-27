"""Process-wide pacing and shared HTTP 429 backoff for HF upload API calls."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter


def is_hub_api(url):
    parsed = urlsplit(url)
    return parsed.hostname in {'huggingface.co', 'hf.co'} and (
        parsed.path.startswith('/api/') or '/info/lfs/' in parsed.path)


def retry_delay(headers, wall_time=None):
    now = time.time() if wall_time is None else wall_time
    value = headers.get('Retry-After', '')
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(value).timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                pass
    durations = re.findall(r'(?:^|;)\s*t=(\d+)', headers.get('RateLimit', ''))
    if durations:
        return max(map(float, durations))
    return 300.0


class RequestGate:
    def __init__(self, interval=0.5, status_path=None, clock=None, sleep=None):
        if interval <= 0:
            raise ValueError('API request interval must be positive')
        self.interval = interval
        self.status_path = Path(status_path) if status_path else None
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.blocked_until = 0.0

    def _status(self, status, seconds=0, reason=None):
        if self.status_path:
            record = {'status': status, 'minimum_api_interval_seconds': self.interval,
                      'scope': 'shared across every upload worker',
                      'updated_at': datetime.now(timezone.utc).isoformat()}
            if seconds:
                record['retry_not_before'] = (
                    datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()
            if reason:
                record['reason'] = reason
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(self.status_path)

    def wait(self):
        while True:
            with self.lock:
                now = self.clock()
                deadline = max(self.next_request, self.blocked_until)
                if deadline <= now:
                    self.next_request = now + self.interval
                    if self.blocked_until:
                        self.blocked_until = 0.0
                        self._status('resumed_with_api_pacing')
                    return
                delay = deadline - now
            # Never hold the lock while sleeping; another worker may extend backoff.
            self.sleep(min(delay, 30.0))

    def cooldown(self, seconds, reason='HTTP 429'):
        with self.lock:
            self.blocked_until = max(self.blocked_until, self.clock() + seconds)
            remaining = self.blocked_until - self.clock()
            self._status('waiting_for_rate_limit_reset', remaining, reason)
        print(f'HF API backoff: {remaining:.0f}s ({reason}); completed upload metadata retained.',
              flush=True)


class RateLimitedAdapter(HTTPAdapter):
    def __init__(self, gate, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate

    def send(self, request, **kwargs):
        if not is_hub_api(request.url):
            return super().send(request, **kwargs)
        for attempt in range(6):
            self.gate.wait()
            response = super().send(request, **kwargs)
            if response.status_code != 429 or attempt == 5:
                return response
            delay = retry_delay(response.headers) + 2.0
            response.close()
            self.gate.cooldown(delay)


def configure_upload_backend(interval=0.5, initial_cooldown=0, status_path=None):
    from huggingface_hub import configure_http_backend
    gate = RequestGate(interval, status_path)
    if initial_cooldown:
        gate.cooldown(initial_cooldown, 'cooldown after the previous rate-limited upload')
    def factory():
        session = requests.Session()
        session.mount('https://', RateLimitedAdapter(gate, pool_maxsize=8))
        session.mount('http://', RateLimitedAdapter(gate, pool_maxsize=8))
        return session
    configure_http_backend(backend_factory=factory)
    return gate
