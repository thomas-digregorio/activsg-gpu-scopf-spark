from activsg_scopf.commitment import commitment_snapshot


def _solution(commitments: list[int]) -> dict[str, object]:
    return {
        "generators": [
            {
                "source_id": f"gen-row-{index + 1:04d}",
                "source_row": index + 1,
                "bus": 100 + index,
                "commitment": float(value),
            }
            for index, value in enumerate(commitments)
        ]
    }


def test_round_commitment_snapshot_tracks_source_row_flips() -> None:
    first = commitment_snapshot(_solution([1, 0, 1]), None)
    second = commitment_snapshot(_solution([1, 1, 0]), first)
    stable = commitment_snapshot(_solution([1, 1, 0]), second)

    assert first["commitment_count"] == 2
    assert first["stable_from_previous_round"] is None
    assert second["commitment_count"] == 2
    assert second["hamming_distance_from_previous_round"] == 2
    assert second["off_to_on_from_previous_round"] == 1
    assert second["on_to_off_from_previous_round"] == 1
    assert [
        record["source_id"]
        for record in second["changes_from_previous_round"]
    ] == ["gen-row-0002", "gen-row-0003"]
    assert stable["stable_from_previous_round"] is True
    assert stable["hamming_distance_from_previous_round"] == 0
    assert (
        stable["commitment_fingerprint_sha256"]
        == second["commitment_fingerprint_sha256"]
    )
