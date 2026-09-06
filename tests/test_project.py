from pathlib import Path


def test_required_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "src/app.py",
        "src/crew.py",
        "src/rag.py",
        "src/governance.py",
        "src/guardrails.py",
        "src/schemas.py",
    ]

    for file in required_files:
        assert Path(file).exists(), f"Missing required file: {file}"


def test_project_structure():
    assert Path("src").is_dir()
    assert Path("tests").is_dir()
