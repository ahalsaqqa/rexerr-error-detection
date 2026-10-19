#!/usr/bin/env bash
# Download ReXErr-v1 (open access, ODC-By 1.0) from PhysioNet and verify the checksums.
# usage: bash scripts/download_data.sh [DATA_DIR]
set -euo pipefail
DATA_DIR=${1:-data}
URL=https://physionet.org/content/rexerr-v1/get-zip/1.0.0/
mkdir -p "$DATA_DIR"
# PhysioNet often resets long downloads; curl -C - resumes where it stopped.
for i in $(seq 1 40); do
  curl -sS -L -C - -o "$DATA_DIR/rexerr.zip" "$URL" && break
  echo "download interrupted, retrying ($i)"; sleep 5
done
unzip -q -o "$DATA_DIR/rexerr.zip" -d "$DATA_DIR/.unzip"
cp -r "$DATA_DIR"/.unzip/*/* "$DATA_DIR"/
rm -rf "$DATA_DIR/.unzip"
(cd "$DATA_DIR" && sha256sum -c --quiet SHA256SUMS.txt && echo "ReXErr-v1 ready in $DATA_DIR")
