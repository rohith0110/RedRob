from pathlib import Path


def test_sandbox_ranks_demo_candidates():
    from app import rank_uploaded_candidates

    demo_path = Path("sample_data/demo_candidates.jsonl")
    result = rank_uploaded_candidates(demo_path.read_bytes(), demo_path.name)

    assert result["rows"]
    assert len(result["rows"]) <= 100
    assert result["download_csv"].startswith("candidate_id,rank,score,reasoning")
    assert all("reasoning" in row and row["reasoning"] for row in result["rows"])
