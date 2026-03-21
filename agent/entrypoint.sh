#!/bin/sh
# Generate TLS cert before starting uvicorn
python3 -c "
import main
# The module-level code in main.py already generates the cert
print('TLS cert ready at', main._TLS_CERT_PATH)
"
exec uvicorn main:app --host 0.0.0.0 --port 8090 \
  --ssl-keyfile /data/tls/key.pem \
  --ssl-certfile /data/tls/cert.pem
