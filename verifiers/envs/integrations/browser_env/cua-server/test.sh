#!/bin/bash

# CUA Primitives API Server - Canary Test
#
# This script tests the basic functionality of the CUA server.
# Run after starting the server with ./start.sh
#
# Usage:
#   ./test.sh                     # Test against localhost:3000
#   ./test.sh http://localhost:8080  # Custom server URL

set -e

BASE_URL="${1:-http://localhost:3000}"

echo "============================================"
echo "CUA Primitives API Server - Canary Test"
echo "============================================"
echo "Server: $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
  echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail() {
  echo -e "${RED}✗ FAIL${NC}: $1"
  exit 1
}

info() {
  echo -e "${YELLOW}→${NC} $1"
}

# Test 1: Health check
info "Testing health endpoint..."
HEALTH=$(curl -s "$BASE_URL/health")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
  pass "Health check"
else
  fail "Health check - unexpected response: $HEALTH"
fi

# Test 2: List sessions (should be empty or have sessions)
info "Testing list sessions..."
SESSIONS=$(curl -s "$BASE_URL/sessions")
if echo "$SESSIONS" | grep -q '"sessions"'; then
  pass "List sessions"
else
  fail "List sessions - unexpected response: $SESSIONS"
fi

# Test 3: Create a session
info "Creating browser session..."
CREATE_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{"env": "LOCAL"}')

SESSION_ID=$(echo "$CREATE_RESPONSE" | grep -o '"sessionId":"[^"]*"' | cut -d'"' -f4)

if [ -z "$SESSION_ID" ]; then
  fail "Create session - no sessionId in response: $CREATE_RESPONSE"
fi

if echo "$CREATE_RESPONSE" | grep -q '"screenshot"'; then
  pass "Create session (ID: $SESSION_ID)"
else
  fail "Create session - missing screenshot in response"
fi

# Test 4: Get session state
info "Getting session state..."
STATE=$(curl -s "$BASE_URL/sessions/$SESSION_ID/state")
if echo "$STATE" | grep -q '"screenshot"'; then
  pass "Get session state"
else
  fail "Get session state - unexpected response: $STATE"
fi

# Test 5: Navigate to Stagehand login eval site (has input fields for type test)
info "Testing goto action..."
GOTO_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "goto", "url": "https://browserbase.github.io/stagehand-eval-sites/sites/login/"}')

if echo "$GOTO_RESPONSE" | grep -q '"success":true'; then
  pass "Goto action"
else
  fail "Goto action - unexpected response: $GOTO_RESPONSE"
fi

# Verify URL changed
if echo "$GOTO_RESPONSE" | grep -q 'stagehand-eval-sites'; then
  pass "URL updated to Stagehand eval site"
else
  fail "URL not updated - response: $GOTO_RESPONSE"
fi

# Test 6: Click on email input field (centered form in 1024x768 viewport)
info "Testing click on email field..."
CLICK_EMAIL_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "click", "x": 512, "y": 310}')

if echo "$CLICK_EMAIL_RESPONSE" | grep -q '"success":true'; then
  pass "Click email field"
else
  fail "Click email field - unexpected response: $CLICK_EMAIL_RESPONSE"
fi

# Wait after click for focus
info "Waiting after click..."
curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 1000}' > /dev/null

# Test 7: Type email address
info "Testing type action (email)..."
TYPE_EMAIL_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "type", "text": "test@example.com"}')

if echo "$TYPE_EMAIL_RESPONSE" | grep -q '"success":true'; then
  pass "Type email"
else
  fail "Type email - unexpected response: $TYPE_EMAIL_RESPONSE"
fi

# Wait after typing
info "Waiting after type..."
curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 1000}' > /dev/null

# Test 8: Tab to password field
info "Testing Tab to password field..."
TAB_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "keypress", "keys": "Tab"}')

if echo "$TAB_RESPONSE" | grep -q '"success":true'; then
  pass "Tab to password field"
else
  fail "Tab to password field - unexpected response: $TAB_RESPONSE"
fi

# Wait after tab
curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 1000}' > /dev/null

# Test 9: Type password
info "Testing type action (password)..."
TYPE_PASSWORD_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "type", "text": "secretpassword123"}')

if echo "$TYPE_PASSWORD_RESPONSE" | grep -q '"success":true'; then
  pass "Type password"
else
  fail "Type password - unexpected response: $TYPE_PASSWORD_RESPONSE"
fi

# Wait after typing password
info "Waiting after password..."
curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 1000}' > /dev/null

# Test 10: Click Sign in button (centered in 1024x768 viewport)
info "Testing click Sign in button..."
CLICK_SIGNIN_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "click", "x": 512, "y": 500}')

if echo "$CLICK_SIGNIN_RESPONSE" | grep -q '"success":true'; then
  pass "Click Sign in button"
else
  fail "Click Sign in button - unexpected response: $CLICK_SIGNIN_RESPONSE"
fi

# Wait after sign in click
info "Waiting after Sign in..."
curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 2000}' > /dev/null

# Test 11: Scroll action
info "Testing scroll action..."
SCROLL_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "scroll", "x": 512, "y": 384, "scroll_y": 100}')

if echo "$SCROLL_RESPONSE" | grep -q '"success":true'; then
  pass "Scroll action"
else
  fail "Scroll action - unexpected response: $SCROLL_RESPONSE"
fi

# Test 12: Wait action
info "Testing wait action..."
WAIT_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions/$SESSION_ID/action" \
  -H "Content-Type: application/json" \
  -d '{"type": "wait", "timeMs": 500}')

if echo "$WAIT_RESPONSE" | grep -q '"success":true'; then
  pass "Wait action"
else
  fail "Wait action - unexpected response: $WAIT_RESPONSE"
fi

# Test 13: Delete session
info "Deleting session..."
DELETE_RESPONSE=$(curl -s -X DELETE "$BASE_URL/sessions/$SESSION_ID")

if echo "$DELETE_RESPONSE" | grep -q '"success":true'; then
  pass "Delete session"
else
  fail "Delete session - unexpected response: $DELETE_RESPONSE"
fi

# Test 14: Verify session is gone
info "Verifying session deleted..."
GONE_RESPONSE=$(curl -s "$BASE_URL/sessions/$SESSION_ID/state")

if echo "$GONE_RESPONSE" | grep -q 'SESSION_NOT_FOUND'; then
  pass "Session properly deleted"
else
  fail "Session still exists after deletion"
fi

echo ""
echo "============================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "============================================"

