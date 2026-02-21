#!/bin/bash
# Deployment script for contest UI to new HuggingFace Space

set -e

echo "🚀 Deploying Contest UI to HuggingFace Space"
echo "============================================="

# Check if we're on the contest branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "contest-ui-refactor" ]; then
    echo "⚠️  Not on contest-ui-refactor branch (currently on: $CURRENT_BRANCH)"
    echo "Switching to contest-ui-refactor..."
    git checkout contest-ui-refactor
fi

# Show current branch and latest commit
echo ""
echo "📍 Branch: $(git branch --show-current)"
echo "📝 Latest commit:"
git log -1 --oneline

# Prompt for HuggingFace Space name
echo ""
read -p "Enter HuggingFace Space name (e.g., SqlQueryBuddyContest): " SPACE_NAME

if [ -z "$SPACE_NAME" ]; then
    echo "❌ Space name cannot be empty"
    exit 1
fi

# Prompt for username
echo ""
read -p "Enter your HuggingFace username: " HF_USERNAME

if [ -z "$HF_USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

# Construct HuggingFace Space URL
HF_SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
HF_GIT_URL="${HF_SPACE_URL%.git}.git"

echo ""
echo "📦 Target Space: $HF_SPACE_URL"
echo ""

# Check if remote already exists
if git remote | grep -q "^hf-contest$"; then
    echo "⚠️  Remote 'hf-contest' already exists. Removing..."
    git remote remove hf-contest
fi

# Add new remote
echo "➕ Adding remote: hf-contest"
git remote add hf-contest "$HF_GIT_URL"

# Push to HuggingFace
echo ""
echo "🚀 Pushing to HuggingFace Space..."
echo "   (You may be prompted for HuggingFace credentials)"
echo ""

git push hf-contest contest-ui-refactor:main --force

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Visit your Space at: $HF_SPACE_URL"
echo ""
echo "⚠️  Note: First deployment may take 2-3 minutes to build."
echo "   Check build logs at: ${HF_SPACE_URL}/logs"
echo ""
echo "📊 Contest Features Deployed:"
echo "   • Agent loop visualization with real-time timing"
echo "   • Accordion-based single-screen layout"
echo "   • All 6 steps (Query → RAG → SQL → Validate → Execute → Insights)"
echo ""
