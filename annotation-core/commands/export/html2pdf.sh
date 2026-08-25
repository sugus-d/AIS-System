#!/bin/bash
INPUT="$1"; OUTPUT="$2"; CSS="${3:-/tmp/cjk-html.css}"
[ -z "$INPUT" ] && echo "Usage: $0 input.md output.pdf" && exit 1
[ -z "$OUTPUT" ] && OUTPUT="${INPUT%.md}.pdf"
TMPHTML=$(mktemp /tmp/html2pdf_XXXXX.html)
pandoc "$INPUT" -o "$TMPHTML" --standalone --mathml --metadata title=" " 2>/dev/null
[ -f "$CSS" ] && sed -i "s|</head>|<link rel=\"stylesheet\" href=\"$CSS\"></head>|" "$TMPHTML"
/usr/local/bin/chromium --headless --no-sandbox --disable-gpu \
    --print-to-pdf="$OUTPUT" --virtual-time-budget=10000 \
    --no-margins \
    "file://$TMPHTML" 2>/dev/null
rm -f "$TMPHTML"
echo "PDF: $OUTPUT"
