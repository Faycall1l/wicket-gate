import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.K6_RATE) || 20,  // default 20 rps; override via K6_RATE
      timeUnit: '1s',
      duration: __ENV.K6_TEST_DURATION || '5m',  // custom name on purpose:
      // k6 reserves K6_DURATION (and --duration) and drops scenarios if used.
      // thresholds abortOnFail by default: pass --no-thresholds for raw capture.
      preAllocatedVUs: 10,
      maxVUs: 50,
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500'],
    errors: ['rate<0.01'],
  },
  summaryTrendStats: ['min', 'avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const lat = new Trend('req_lat_ms');
const errors = new Rate('errors');
const BASE = __ENV.K6_TARGET_URL || 'http://localhost:18080/';

export default function () {
  const res = http.get(`${BASE}`);
  lat.add(res.timings.duration);
  errors.add(res.status !== 200);
  check(res, { 'status is 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  const m = data.metrics;
  const get = (k) => (m[k] ? m[k].values : {});
  return {
    stdout: JSON.stringify({
      requests: get('http_reqs').count,
      success_rate: 1 - (get('errors').rate || 0),
      p50_ms: get('http_req_duration')['med'],
      p95_ms: get('http_req_duration')['p(95)'],
      p99_ms: get('http_req_duration')['p(99)'],
      max_ms: get('http_req_duration')['max'],
    }, null, 2),
  };
}
