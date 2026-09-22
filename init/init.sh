#!/bin/bash
set -euo pipefail

TOKEN_FILE=/internal/admin-token.txt
DS_FILE=/internal/influxdb3.yaml
ENV_FILE=/hostenv/.env

echo ">> Creating InfluxDB3 admin token..."

if [ -s "$TOKEN_FILE" ]; then
  echo ">> Existing token found, checking if it's still valid..."
  CANDIDATE_TOKEN=$(cat "$TOKEN_FILE")
  if curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $CANDIDATE_TOKEN" "$INFLUXDB_HOST/ping" | grep -q '^200$'; then
    echo ">> Token is valid, reusing it."
    TOKEN="$CANDIDATE_TOKEN"
  else
    echo ">> Saved token is no longer valid (server was likely reset). Regenerating..."
    rm -f "$TOKEN_FILE"
  fi
fi

if [ ! -s "$TOKEN_FILE" ]; then
  RAW_OUTPUT=$(influxdb3 create token --admin --host "$INFLUXDB_HOST" --format json 2>&1) || {
    echo "$RAW_OUTPUT"
    echo "ERROR: could not create the token."
    exit 1
  }
  echo "$RAW_OUTPUT"

  TOKEN=$(echo "$RAW_OUTPUT" | grep -o '"token"[[:space:]]*:[[:space:]]*"[^"]*"' | sed -E 's/.*"([^"]+)"$/\1/')
  if [ -z "$TOKEN" ]; then
    TOKEN=$(echo "$RAW_OUTPUT" | grep -Eo '[A-Za-z0-9_\-]{40,}' | head -n1)
  fi
  if [ -z "$TOKEN" ]; then
    echo "ERROR: could not extract the token from the output above."
    exit 1
  fi

  echo -n "$TOKEN" > "$TOKEN_FILE"
fi

echo ">> Token ready: $(cat $TOKEN_FILE | cut -c1-8)..."

echo ">> Writing .env file (host-visible) with the token..."
cat > "$ENV_FILE" <<ENVEOF
INFLUXDB_TOKEN=${TOKEN}
INFLUXDB_HOST=${INFLUXDB_HOST}
INFLUXDB_DB=${INFLUX_DB_NAME}
ENVEOF

echo ">> Creating database '${INFLUX_DB_NAME}' (if it doesn't already exist)..."
influxdb3 create database --host "$INFLUXDB_HOST" --token "$TOKEN" "$INFLUX_DB_NAME" || \
  echo ">> Database probably already exists, continuing."

echo ">> Generating Grafana datasource provisioning file (internal, Docker-managed)..."
cat > "$DS_FILE" <<INNEREOF
apiVersion: 1
datasources:
  - name: InfluxDB3
    type: influxdb
    access: proxy
    url: ${INFLUXDB_HOST}
    isDefault: true
    jsonData:
      version: SQL
      httpMode: POST
      dbName: ${INFLUX_DB_NAME}
      insecureGrpc: true
    secureJsonData:
      token: ${TOKEN}
INNEREOF

echo ">> Done. Grafana will start with the InfluxDB3 datasource already configured."