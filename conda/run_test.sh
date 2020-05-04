#!/bin/bash

set -e

VERSION=$(cat VERSION)

export PYTHONPATH=''

${PYTHON} <<EOF
print("="*80)
print("\trun_test.sh for Xerus ${VERSION}")
print("="*80)
import xerus
assert xerus.__version__ == '${VERSION}', xerus.__version__+" != ${VERSION}"
EOF

exit 0
