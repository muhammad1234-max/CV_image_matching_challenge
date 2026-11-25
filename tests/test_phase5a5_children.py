"""Exercise generated Kaggle children with external native services supplied by the CPU fixture."""
import pytest
import phase5a3_cpu
from test_phase5a3 import lab
from src.runner_5a3 import colab
from src.runner_5a5.commands import prepare
from src.runner_5a1.contract import Fatal,STAGES,read

@pytest.mark.parametrize("mode",["complete","timeout","digest"])
def test_platform_chain(lab,monkeypatch,mode):
    fx,cpu,run=lab
    namespace=phase5a3_cpu.NS
    def native_namespace(**kwargs):
        if "get_device_name" in kwargs:kwargs["device_count"]=lambda:1
        return namespace(**kwargs)
    monkeypatch.setattr(phase5a3_cpu,"NS",native_namespace)
    def generate(session,stage,work,d,root,contract):
        return prepare(session,stage,work,d,root,contract,fx.tmp/"ns64_clean",
            fx.tmp/(colab.DS+"_opaque.zip"),"Tesla T4, 580.82.07, 15360 MiB",{})
    monkeypatch.setattr(colab,"generate_command",generate)
    for stage in STAGES:
        if stage=="trial":
            cpu.failure=mode
            if mode=="digest":
                with pytest.raises(Fatal):run(stage)
                assert not (fx.r.root/"decision").exists()
                return
        run(stage)
        assert fx.r.verify(stage,fx.r.root/stage)
    assert read(fx.r.root/"decision/selection.json")["action"]==("ABSTAIN" if mode=="timeout" else "REJECT")
