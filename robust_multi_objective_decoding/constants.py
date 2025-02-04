import pathlib

ProjectDir = pathlib.Path(__file__).parent.parent
DatasetDir = ProjectDir / "datasets"
DatasetDir.mkdir(exist_ok=True)
