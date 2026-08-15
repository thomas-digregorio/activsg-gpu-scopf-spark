"""Exercise cuOpt's native MIP-start path on a tiny deterministic MILP."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presolve", choices=("default", "off"), required=True)
    parser.add_argument("--log-file")
    args = parser.parse_args()

    from cuopt.linear_programming.problem import (
        CONTINUOUS,
        INTEGER,
        MINIMIZE,
        Problem,
    )
    from cuopt.linear_programming.solver_settings import SolverSettings

    problem = Problem("mip_start_contract_probe")
    cheap = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name="cheap")
    expensive = problem.addVariable(
        lb=0.0, ub=1.0, vtype=INTEGER, name="expensive"
    )
    dispatch = problem.addVariable(
        lb=0.0, ub=6.0, vtype=CONTINUOUS, name="dispatch"
    )
    problem.addConstraint(dispatch >= 5.0, name="demand")
    problem.addConstraint(
        dispatch <= 4.0 * cheap + 6.0 * expensive,
        name="capacity",
    )
    problem.setObjective(
        10.0 * cheap + 20.0 * expensive + dispatch,
        sense=MINIMIZE,
    )
    cheap.setMIPStart(0.0)
    expensive.setMIPStart(1.0)

    settings = SolverSettings()
    settings.set_parameter("log_to_console", True)
    settings.set_parameter("random_seed", 0)
    settings.set_parameter("mip_relative_gap", 0.0)
    settings.set_parameter("time_limit", 10.0)
    if args.log_file:
        settings.set_parameter("log_file", args.log_file)
    if args.presolve == "off":
        settings.set_parameter("presolve", 0)
    problem._to_data_model()
    initial_primal = problem.model.get_initial_primal_solution()
    print(
        json.dumps(
            {
                "initial_primal_shape": list(initial_primal.shape),
                "initial_primal": initial_primal.tolist(),
            },
            sort_keys=True,
        )
    )
    problem.solve(settings)
    print(
        json.dumps(
            {
                "presolve": args.presolve,
                "status": problem.Status.name,
                "objective": float(problem.ObjValue),
                "values": {
                    variable.VariableName: float(variable.getValue())
                    for variable in problem.getVariables()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
