#!/bin/bash
# Reset llm-manager UAT database with seed data.
# Connects via Cloud SQL proxy on the UAT pod.
#
# Usage: ./scripts/reset-uat.sh
#
# Requires: kubectl access to the llm-manager namespace

set -euo pipefail

NAMESPACE="llm-manager"
POD=$(kubectl get pods -n "$NAMESPACE" -l app=llm-manager-backend-uat -o jsonpath='{.items[0].metadata.name}')
DB_NAME="llm_manager_uat"

if [ -z "$POD" ]; then
  echo "ERROR: No llm-manager-backend-uat pod found"
  exit 1
fi

# Safety check — refuse to run on non-UAT databases
echo "Target pod: $POD"
echo "Target database: $DB_NAME"

# Get the DATABASE_URL from the pod to verify it's UAT
DB_URL=$(kubectl exec -n "$NAMESPACE" "$POD" -c backend -- printenv DATABASE_URL 2>/dev/null || true)
if [[ "$DB_URL" != *"uat"* ]]; then
  echo "ERROR: DATABASE_URL does not contain 'uat' — refusing to run"
  echo "Got: $DB_URL"
  exit 1
fi

echo "Verified UAT database. Resetting..."

# Port-forward the cloud-sql-proxy port
kubectl port-forward -n "$NAMESPACE" "$POD" 15432:5432 &
PF_PID=$!
sleep 3

# Extract user and password from DATABASE_URL
DB_USER=$(echo "$DB_URL" | sed -n 's|postgresql://\([^:]*\):.*|\1|p')
DB_PASS=$(echo "$DB_URL" | sed -n 's|postgresql://[^:]*:\([^@]*\)@.*|\1|p')

export PGPASSWORD="$DB_PASS"

run_sql() {
  psql -h 127.0.0.1 -p 15432 -U "$DB_USER" -d "$DB_NAME" -q -c "$1"
}

echo "Truncating all tables..."
run_sql "
TRUNCATE TABLE
  app_rate_limits,
  app_allowed_models,
  queue_jobs,
  profile_activations,
  profile_image_entries,
  profile_model_entries,
  model_settings,
  cloud_model_config,
  api_keys,
  ollama_library_cache,
  library_cache_meta,
  model_safety_tags,
  profiles,
  registered_apps,
  llm_runners,
  llm_agents
CASCADE;
"

echo "Seeding profiles..."
run_sql "
INSERT INTO profiles (name, is_default, unsafe_enabled) VALUES
  ('Default', true, false),
  ('Creative', false, true),
  ('Safe Only', false, false);
"

echo "Seeding safety tags..."
run_sql "
INSERT INTO model_safety_tags (pattern, classification, reason) VALUES
  ('*uncensored*', 'unsafe', 'Model trained without safety restrictions'),
  ('dolphin-*', 'unsafe', 'Dolphin models are uncensored by design'),
  ('wizard-vicuna*', 'unsafe', 'WizardVicuna uncensored variant'),
  ('*abliterated*', 'unsafe', 'Model with safety training removed');
"

echo "Seeding runners..."
run_sql "
INSERT INTO llm_runners (hostname, address, port, capabilities, last_seen) VALUES
  ('murderbot', 'http://10.100.20.19', 8090, '{\"gpu\": \"NVIDIA RTX 4090\", \"vram_gb\": 24, \"gpu_vendor\": \"nvidia\"}', NOW()),
  ('archbox', 'http://10.100.20.20', 8090, '{\"gpu\": \"AMD RX 7900 XTX\", \"vram_gb\": 24, \"gpu_vendor\": \"amd\"}', NOW());
"

echo "Seeding apps..."
run_sql "
INSERT INTO registered_apps (name, base_url, api_key, status, allow_profile_switch, metadata) VALUES
  ('ecdysis', 'http://ecdysis-backend:8082', 'uat-key-ecdysis-001', 'active', true, '{\"environment\": \"uat\"}'),
  ('home-assistant', 'http://ha.amer.dev', 'uat-key-ha-001', 'active', false, '{\"environment\": \"uat\"}'),
  ('dev-notebook', NULL, 'uat-key-notebook-001', 'active', true, '{\"environment\": \"uat\"}');
"

echo "Seeding app allowed models..."
run_sql "
INSERT INTO app_allowed_models (app_id, model_pattern)
SELECT id, pattern FROM registered_apps
CROSS JOIN (VALUES ('qwen*'), ('llama*'), ('mistral*')) AS p(pattern)
WHERE name = 'ecdysis';

INSERT INTO app_allowed_models (app_id, model_pattern)
SELECT id, '*' FROM registered_apps WHERE name = 'home-assistant';

INSERT INTO app_allowed_models (app_id, model_pattern)
SELECT id, '*' FROM registered_apps WHERE name = 'dev-notebook';
"

echo "Seeding profile model entries..."
run_sql "
INSERT INTO profile_model_entries (profile_id, model_safe, model_unsafe, count, label, sort_order)
SELECT p.id, m.safe, m.unsafe, m.cnt, m.lbl, m.ord
FROM profiles p
CROSS JOIN (VALUES
  ('qwen2.5:7b', NULL, 1, 'General Chat', 0),
  ('llama3.2:3b', NULL, 1, 'Fast Answers', 1),
  ('mistral:7b', NULL, 1, 'Code Assistant', 2)
) AS m(safe, unsafe, cnt, lbl, ord)
WHERE p.name = 'Default';

INSERT INTO profile_model_entries (profile_id, model_safe, model_unsafe, count, label, sort_order)
SELECT p.id, m.safe, m.unsafe, m.cnt, m.lbl, m.ord
FROM profiles p
CROSS JOIN (VALUES
  ('qwen2.5:7b', 'dolphin-qwen2.5:7b', 1, 'General Chat', 0),
  ('llama3.2:3b', 'dolphin-llama3:8b', 1, 'Unrestricted', 1)
) AS m(safe, unsafe, cnt, lbl, ord)
WHERE p.name = 'Creative';
"

echo "Seeding model settings..."
run_sql "
INSERT INTO model_settings (model_name, do_not_evict, evictable, vram_estimate_gb) VALUES
  ('qwen2.5:7b', false, true, 4.5),
  ('llama3.2:3b', false, true, 2.0),
  ('mistral:7b', false, true, 4.5);
"

echo "Seeding app rate limits..."
run_sql "
INSERT INTO app_rate_limits (app_id, max_queue_depth, max_jobs_per_minute)
SELECT id, depth, rpm FROM registered_apps r
JOIN (VALUES
  ('ecdysis', 100, 20),
  ('home-assistant', 20, 5),
  ('dev-notebook', 50, 10)
) AS l(name, depth, rpm) ON r.name = l.name;
"

# Cleanup
kill $PF_PID 2>/dev/null || true
wait $PF_PID 2>/dev/null || true

echo ""
echo "=== UAT database reset complete ==="
echo "  Profiles: Default, Creative, Safe Only"
echo "  Runners: murderbot (NVIDIA), archbox (AMD)"
echo "  Apps: ecdysis, home-assistant, dev-notebook"
echo "  Models: qwen2.5:7b, llama3.2:3b, mistral:7b"
