#!/bin/sh
# n1os for Linux - Strata's ./setup.sh name for n1os installer (./n1os.sh): the first run sets everything up
# and starts the dashboard; later runs just start it.  ./setup.sh --help lists the options.
cd "$(dirname "$0")" || exit 1
exec ./n1os.sh "$@"
