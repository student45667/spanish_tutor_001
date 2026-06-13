#!/bin/bash

# Usage: ./git_push.sh "commit message"
# If no message provided, uses a default timestamp message

MSG=${1:-"update $(date '+%Y-%m-%d %H:%M:%S')"}

git add -A
git commit -m "$MSG"
git push origin main
