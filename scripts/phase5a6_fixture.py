"""Isolated native-service fixture for generated stage programs."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

def main():
    spec=json.loads(Path(sys.argv[1]).read_bytes())
    kit=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(kit),str(kit/"tests")]
    import phase5a3_cpu as native
    from src.novel_model.final_candidate_v1 import workflow, session
    session.KIT=Path(spec["base"])/"kit"
    assert Path(workflow.__file__).is_relative_to(kit)
    assert Path(sys.executable)==Path(spec["python"])
    assert os.environ["CUDA_VISIBLE_DEVICES"]==spec["visible"]
    def namespace(**values):
        if "get_device_name" in values:
            values["device_count"]=lambda:1
        return SimpleNamespace(**values)
    cpu=native.CPU.__new__(native.CPU)
    cpu.base=Path(spec["base"])
    cpu.original=cpu.base/"dbfixture"
    cpu.fx=SimpleNamespace(tmp=cpu.base,names=spec["names"])
    cpu.mem=cpu.base/"meminfo"
    cpu.p=spec["pins"]
    cpu.failure=spec["fault"]
    with patch.object(native,"NS",namespace):
        try:
            cpu.monitor(spec["argv"],Path(spec["work"]),Path(spec["dest"]),spec["end"],spec["stage"])
        except TimeoutError:
            sys.exit(124)
if __name__=="__main__":
    main()
