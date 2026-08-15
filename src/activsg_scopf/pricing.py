"""Fixed-commitment preventive SCOPF pricing for completed MILP runs."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .errors import ScopfError
from .matpower import GEN_BUS, MatpowerCase
from .model import MasterModel
from .network import ContingencyCatalog, NetworkData
from .screening import ContingencyScreener, add_security_pairs
from .solution import serialize_solution
from .solvers.highs import HighsSession


def _price_summary(prices: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(prices)),
        "p01": float(np.quantile(prices, 0.01)),
        "p05": float(np.quantile(prices, 0.05)),
        "median": float(np.median(prices)),
        "mean": float(np.mean(prices)),
        "p95": float(np.quantile(prices, 0.95)),
        "p99": float(np.quantile(prices, 0.99)),
        "maximum": float(np.max(prices)),
        "standard_deviation": float(np.std(prices)),
    }


def run_fixed_commitment_pricing(
    case: MatpowerCase,
    network: NetworkData,
    catalog: ContingencyCatalog,
    master: MasterModel,
    mip_values: np.ndarray,
    *,
    already_added_pair_ids: set[str],
    security_tolerance_pu: float,
    screen_chunk_columns: int,
    solver_threads: int,
    maximum_rounds: int,
) -> dict[str, Any]:
    """Fix the MILP commitment, solve a secure LP, and return nodal dual prices.

    The LP starts with the final MILP restricted master. Its dispatch can differ
    from the MIP incumbent, so it is exhaustively screened and receives any new
    violated contingency rows before prices are accepted.
    """

    started = time.perf_counter()
    pricing_master = copy.deepcopy(master.canonical)
    commitment_columns = np.asarray(
        sorted(master.index.commitment_by_generator.values()), dtype=np.int32
    )
    raw_commitments = np.asarray(mip_values[commitment_columns], dtype=np.float64)
    fixed_commitments = np.rint(raw_commitments)
    maximum_commitment_rounding = float(
        np.max(np.abs(raw_commitments - fixed_commitments), initial=0.0)
    )
    if maximum_commitment_rounding > 1e-5:
        raise ScopfError(
            "Cannot price a solution whose commitment values are not integral: "
            f"maximum rounding residual {maximum_commitment_rounding}"
        )
    for column, fixed in zip(commitment_columns, fixed_commitments, strict=True):
        pricing_master.integrality[int(column)] = 0
        pricing_master.column_lower[int(column)] = float(fixed)
        pricing_master.column_upper[int(column)] = float(fixed)

    screener = ContingencyScreener(
        network,
        catalog,
        backend="numpy",
        chunk_columns=screen_chunk_columns,
    )
    session = HighsSession(
        pricing_master,
        mip_relative_gap=0.0,
        threads=solver_threads,
    )
    known_pairs = set(already_added_pair_ids)
    pricing_added_pairs: list[str] = []
    rounds: list[dict[str, Any]] = []
    final_result = None
    final_screen = None
    for round_number in range(1, maximum_rounds + 1):
        solve_started = time.perf_counter()
        result = session.solve(time_limit_seconds=None)
        solve_wall = time.perf_counter() - solve_started
        if not result.optimal or result.values is None:
            raise ScopfError(
                f"Fixed-commitment pricing LP round {round_number} ended as {result.status}"
            )
        flow = result.values[master.index.flow_by_active_branch]
        screen_started = time.perf_counter()
        screened = screener.screen(
            np.asarray(flow, dtype=np.float64),
            tolerance_pu=security_tolerance_pu,
            already_added=known_pairs,
        )
        screen_wall = time.perf_counter() - screen_started
        new_pair_ids = [pair.pair_id for pair in screened.violations]
        rounds.append(
            {
                "round": round_number,
                "rows_before_solve": pricing_master.num_rows,
                "solve_wall_time_seconds": solve_wall,
                "solver_status": result.status,
                "objective": result.objective,
                "maximum_model_row_violation_pu": (
                    pricing_master.max_row_violation(result.values) / case.base_mva
                ),
                "screen_wall_time_seconds": screen_wall,
                "evaluated_sides": screened.evaluated_pairs,
                "new_violated_pairs": len(new_pair_ids),
                "maximum_security_violation_pu": screened.maximum_violation_pu,
                "maximum_security_pair_id": screened.maximum_pair_id,
                "added_pair_ids": new_pair_ids,
            }
        )
        if not screened.violations:
            if screened.maximum_violation_pu > security_tolerance_pu:
                raise ScopfError(
                    "Fixed-commitment pricing LP violates an already enforced "
                    "security pair beyond tolerance"
                )
            final_result = result
            final_screen = screened
            break
        add_security_pairs(
            pricing_master,
            master.index,
            network,
            screened.violations,
        )
        known_pairs.update(new_pair_ids)
        pricing_added_pairs.extend(new_pair_ids)
    else:
        raise ScopfError("Fixed-commitment pricing reached its constraint-generation limit")

    if final_result is None or final_result.values is None or final_screen is None:
        raise ScopfError("Fixed-commitment pricing did not produce a final solution")
    row_duals = session.last_row_duals
    if row_duals is None or len(row_duals) != pricing_master.num_rows:
        raise ScopfError("HiGHS did not return valid pricing-LP row duals")

    row_lookup = {name: index for index, name in enumerate(pricing_master.row_names)}
    prices_per_mwh = np.asarray(
        [
            row_duals[row_lookup[f"nodal_balance_b{int(bus_id):04d}"]]
            for bus_id in network.bus_ids
        ],
        dtype=np.float64,
    )
    prices_per_pu_hour = prices_per_mwh * case.base_mva
    bus_prices = [
        {
            "bus": int(bus_id),
            "price_per_mwh": float(prices_per_mwh[bus_index]),
            "price_per_pu_hour": float(prices_per_pu_hour[bus_index]),
        }
        for bus_index, bus_id in enumerate(network.bus_ids)
    ]
    price_by_bus = {int(record["bus"]): record for record in bus_prices}

    mip_solution = serialize_solution(mip_values, case, network, master)
    lp_solution = serialize_solution(final_result.values, case, network, master)
    generator_pricing: list[dict[str, Any]] = []
    absolute_dispatch_differences: list[float] = []
    for generator_index, (mip_record, lp_record) in enumerate(
        zip(mip_solution["generators"], lp_solution["generators"], strict=True)
    ):
        bus = int(case.gen[generator_index, GEN_BUS])
        price = price_by_bus[bus]
        difference_mw = float(lp_record["dispatch_mw"] - mip_record["dispatch_mw"])
        absolute_dispatch_differences.append(abs(difference_mw))
        generator_pricing.append(
            {
                "source_id": mip_record["source_id"],
                "source_row": mip_record["source_row"],
                "source_status": mip_record["source_status"],
                "bus": bus,
                "pmin_mw": mip_record["pmin_mw"],
                "pmin_pu": mip_record["pmin_pu"],
                "pmax_mw": mip_record["pmax_mw"],
                "pmax_pu": mip_record["pmax_pu"],
                "commitment": mip_record["commitment"],
                "mip_dispatch_mw": mip_record["dispatch_mw"],
                "mip_dispatch_pu": mip_record["dispatch_pu"],
                "pricing_dispatch_mw": lp_record["dispatch_mw"],
                "pricing_dispatch_pu": lp_record["dispatch_pu"],
                "pricing_minus_mip_dispatch_mw": difference_mw,
                "pricing_minus_mip_dispatch_pu": difference_mw / case.base_mva,
                "nodal_price_per_mwh": price["price_per_mwh"],
                "nodal_price_per_pu_hour": price["price_per_pu_hour"],
            }
        )

    active_security_duals = [
        {
            "pair_id": name,
            "row_dual_per_mw": float(row_duals[index]),
            "row_dual_per_pu_hour": float(row_duals[index] * case.base_mva),
        }
        for index, name in enumerate(pricing_master.row_names)
        if name.startswith("c") and abs(float(row_duals[index])) > 1e-9
    ]
    active_security_duals.sort(key=lambda record: str(record["pair_id"]))
    absolute_differences = np.asarray(absolute_dispatch_differences, dtype=np.float64)
    return {
        "status": "optimal_secure_fixed_commitment_lp",
        "method": (
            "HiGHS continuous LP with the MILP commitment fixed, followed by "
            "exhaustive dynamic branch-N-1 constraint generation"
        ),
        "price_definition": (
            "The nodal-balance row dual is the objective derivative with respect "
            "to one additional MW of demand for the one-hour interval"
        ),
        "price_units": {
            "price_per_mwh": "USD/MWh for the one-hour interval",
            "price_per_pu_hour": (
                "USD per p.u.-hour; price_per_mwh multiplied by baseMVA"
            ),
        },
        "base_mva": case.base_mva,
        "fixed_commitment_count": int(np.count_nonzero(fixed_commitments > 0.5)),
        "maximum_commitment_rounding_residual": maximum_commitment_rounding,
        "mip_objective": float(np.dot(master.canonical.objective, mip_values)),
        "pricing_lp_objective": final_result.objective,
        "mip_minus_pricing_lp_objective": (
            None
            if final_result.objective is None
            else float(np.dot(master.canonical.objective, mip_values) - final_result.objective)
        ),
        "pricing_constraint_generation_rounds": rounds,
        "pricing_added_security_pair_ids": pricing_added_pairs,
        "final_security_pair_count": len(known_pairs),
        "final_exhaustive_violation_pu": final_screen.maximum_violation_pu,
        "final_model_row_violation_pu": (
            pricing_master.max_row_violation(final_result.values) / case.base_mva
        ),
        "pricing_dispatch_difference": {
            "generators_above_1e-6_mw": int(np.count_nonzero(absolute_differences > 1e-6)),
            "sum_absolute_mw": float(np.sum(absolute_differences)),
            "maximum_absolute_mw": float(np.max(absolute_differences, initial=0.0)),
            "sum_absolute_pu": float(np.sum(absolute_differences) / case.base_mva),
            "maximum_absolute_pu": float(
                np.max(absolute_differences, initial=0.0) / case.base_mva
            ),
        },
        "bus_price_summary_per_mwh": _price_summary(prices_per_mwh),
        "bus_price_summary_per_pu_hour": _price_summary(prices_per_pu_hour),
        "bus_prices": bus_prices,
        "generators": generator_pricing,
        "active_security_constraint_duals": active_security_duals,
        "wall_time_seconds": time.perf_counter() - started,
    }
