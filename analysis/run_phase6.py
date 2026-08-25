from __future__ import annotations

from analysis.build_master_results import main as build_master
from analysis.make_core_tables import main as make_tables
from analysis.make_core_figures import main as make_figures
from analysis.make_phase6_report import main as make_report


def main() -> None:
    print("=" * 84)
    print("PHASE 6 — CONSOLIDATED ANALYSIS")
    print("=" * 84)
    build_master()
    make_tables()
    make_figures()
    make_report()
    print("=" * 84)
    print("PHASE 6 COMPLETE")
    print("Outputs:")
    print("  results/phase6/")
    print("  plots/phase6/")
    print("  tables/phase6/")
    print("=" * 84)


if __name__ == "__main__":
    main()
