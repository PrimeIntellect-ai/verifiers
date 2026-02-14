---
name: check-auth
description: Check the users auth status for Prime services
---

To check the users auth status you can:

1. Run `prime config view`
2. Run `prime whoami`
3. Run `cat ~/.prime/config.json`

If the user is not logged in with the Prime CLI, you can do so by running `prime login`

Sometimes problems can occur based on the team the user is setup.  Which can be viewed with the commands above.
Other useful things can be found by running `prime config`
