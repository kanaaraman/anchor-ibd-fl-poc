"""Write one synthetic OMOP CDM extract per hospital site.

    python generate_data.py --out data            # all three sites
    python generate_data.py --out /data --site ous  # one site, e.g. on that site's Spark
    python generate_data.py --out data_eval --eval_n 3000  # large held-out "future patients" cohort per site

The evaluation cohorts exist only because the data is synthetic: they let the
simulator measure each strategy precisely instead of on ~100 test patients.
"""
import argparse
import os

from ibdfl.synth import SITES, write_omop

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--site", choices=list(SITES), action="append")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_n", type=int, help="write an evaluation cohort of this size per site (different patients)")
args = ap.parse_args()

for site in args.site or SITES:
    seed = args.seed + (1000 if args.eval_n else 0)
    counts = write_omop(site, os.path.join(args.out, site), seed=seed, n=args.eval_n)
    s = SITES[site]
    print(f"{site:5s} {s['name']} ({s['registry']}): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
