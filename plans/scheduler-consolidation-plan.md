# Scheduler Consolidation Plan

## Goal
Retire legacy `backend/scheduler.py` and keep a single scheduler implementation (`backend/scheduler_v2.py`) with equivalent runtime behavior for current production workloads.

## Why
- Dual scheduler paths add duplicated logic and subtle divergence risk.
- Lock handling and failure recovery logic already improved in v2.
- Consolidation reduces maintenance cost and incident surface area.

## Phased Approach

### Phase 1: Parity test coverage before removal
- Add/expand tests in:
  - `backend/test_scheduler_v2.py`
  - `backend/test_queue.py`
- Cover parity-critical behaviors:
  - Advisory-lock loss and recovery
  - Job status lifecycle (`queued -> loading_model/running -> terminal`)
  - Queue cancellation race protections
  - Allowed-runner constraints
  - Pinned-runner routing behavior

### Phase 2: Remove legacy flag dependency from runtime wiring
- In `backend/main.py`, remove runtime switching and instantiate only `SimplifiedScheduler`.
- Keep route compatibility surfaces in v2 only where actually needed by callers.

### Phase 3: Delete legacy scheduler implementation
- Remove `backend/scheduler.py` usage from imports/call paths.
- Remove dead compatibility code in v2 once no callers depend on it.

### Phase 4: Operational validation
- Deploy to one environment and monitor:
  - Queue depth progression
  - Dispatch latency
  - Retry/failure rates
  - Lock failover behavior across replicas

## Exit Criteria
- No import/runtime references to `backend/scheduler.py`.
- All scheduler/queue tests pass.
- Production/UAT logs show no stuck queued jobs from scheduler inactivity.
