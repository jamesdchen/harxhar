# DEPRECATED: Use harx.py --segment all instead.
import sys
from src.executor import get_common_parser

from harx import main

if __name__ == '__main__':
    parser = get_common_parser("Segmented Time-Series Backtester (DEPRECATED: use harx.py --segment all)")
    args = parser.parse_args()
    if args.segment is None:
        args.segment = 'all'
    main(args)
