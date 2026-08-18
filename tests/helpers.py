from pathlib import Path

import numpy as np

from activsg_scopf.matpower import ContingencyChange, ContingencyTable, MatpowerCase


def triangle_case() -> tuple[MatpowerCase, ContingencyTable]:
    bus = np.asarray(
        [
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [2, 1, 40, 0, 2, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [3, 1, 20, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        ],
        dtype=float,
    )
    gen = np.asarray(
        [
            [1, 60, 0, 100, -100, 1, 100, 1, 100, 25],
            [3, 0, 0, 100, -100, 1, 100, 0, 50, 10],
        ],
        dtype=float,
    )
    branch = np.asarray(
        [
            [1, 2, 0, 0.1, 0, 100, 0, 0, 0, 0, 1, 0, 0],
            [2, 3, 0, 0.1, 0, 100, 0, 0, 0, 0, 1, 0, 0],
            [1, 3, 0, 0.2, 0, 100, 0, 0, 1.05, 2, 1, 0, 0],
        ],
        dtype=float,
    )
    gencost = np.asarray(
        [
            [2, 0, 0, 3, 0.01, 10, 100],
            [2, 0, 0, 3, 0.02, 12, 80],
        ],
        dtype=float,
    )
    case = MatpowerCase(
        case_name="ACTIVSg500",
        source_path=Path("case_ACTIVSg500.m"),
        sha256="fixture",
        base_mva=100.0,
        bus=bus,
        gen=gen,
        branch=branch,
        gencost=gencost,
    )
    changes = tuple(
        ContingencyChange(i, i, 0, "CT_TBRCH", i, "BR_STATUS", "CT_REP", 0)
        for i in range(1, 4)
    ) + (ContingencyChange(4, 4, 0, "CT_TGEN", 1, "GEN_STATUS", "CT_REP", 0),)
    table = ContingencyTable(Path("contab_ACTIVSg500.m"), "fixture", changes)
    return case, table


def write_triangle_matpower(root: Path) -> tuple[Path, Path]:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    case_path = raw / "case_ACTIVSg500.m"
    case_path.write_text(
        """function mpc = case_ACTIVSg500
mpc.baseMVA = 100;
mpc.bus = [
1 3 0 0 0 0 1 1 0 230 1 1.1 0.9;
2 1 40 0 2 0 1 1 0 230 1 1.1 0.9;
3 1 20 0 0 0 1 1 0 230 1 1.1 0.9;
];
mpc.gen = [
1 60 0 100 -100 1 100 1 100 25;
3 0 0 100 -100 1 100 0 50 10;
];
mpc.branch = [
1 2 0 0.1 0 100 0 0 0 0 1 0 0;
2 3 0 0.1 0 100 0 0 0 0 1 0 0;
1 3 0 0.2 0 100 0 0 1.05 2 1 0 0;
];
mpc.gencost = [
2 0 0 3 0.01 10 100;
2 0 0 3 0.02 12 80;
];
""",
        encoding="utf-8",
    )
    contingency_path = raw / "contab_ACTIVSg500.m"
    contingency_path.write_text(
        """function chgtab = contab_ACTIVSg500
chgtab = [
1 0 CT_TBRCH 1 BR_STATUS CT_REP 0;
2 0 CT_TBRCH 2 BR_STATUS CT_REP 0;
3 0 CT_TBRCH 3 BR_STATUS CT_REP 0;
4 0 CT_TGEN 1 GEN_STATUS CT_REP 0;
];
""",
        encoding="utf-8",
    )
    return case_path, contingency_path
