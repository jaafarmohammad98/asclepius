#!/bin/sh
# Download the public DUTCH sample reports used by test/samples.test.ts.
# They are copyrighted by Precision Analytical, so they are not committed.
set -eu
samples="$(dirname "$0")/../../samples"
mkdir -p "$samples"
cd "$samples"

fetch() {
  mkdir -p "$1"
  [ -f "$1/report.pdf" ] || curl -sSfL -o "$1/report.pdf" "$2"
  echo "ok  $1"
}

fetch male-2019   "https://cdn.shopify.com/s/files/1/0994/7512/files/DUTCH-Complete-Male-Sample-Report-01_24_19-min.pdf"
fetch female-a    "https://cdn.shopify.com/s/files/1/0720/8150/6584/files/DUTCH_Complete_Female_Sample_Report.pdf"
fetch female-2018 "https://cdn.shopify.com/s/files/1/1987/8605/files/DUTCH-Complete-Female-Sample-Report-3_29_18.pdf"
fetch female-2025 "https://fxmed.co.nz/wp-content/uploads/2025/10/DCS-Female-DUTCH-Complete-with-Midsleep.pdf"
