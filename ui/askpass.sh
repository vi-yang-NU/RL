#!/bin/sh
# SSH_ASKPASS helper for the dashboard's Connect button. ssh runs it with the prompt as $1;
# password prompts get the password the dashboard wrote to a private temporary file.
case "$1" in
  *[Pp]assword*)
    if [ -n "$DEEPDISH_ASKPASS_FILE" ] && [ -r "$DEEPDISH_ASKPASS_FILE" ]; then
      cat "$DEEPDISH_ASKPASS_FILE"
      exit 0
    fi
    ;;
esac
echo "deepdish dashboard: unexpected SSH prompt: $1" >&2
exit 1
