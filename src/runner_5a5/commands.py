"""Generate platform-bound children for inspection and fixture testing only."""
import ast
from pathlib import Path
from src.runner_5a3.provenance import command as frozen_command

def once(source,old,new):
    if source.count(old)!=1:raise ValueError("Unexpected frozen platform check")
    return source.replace(old,new)

def prepare(session,stage,work,destination,root,contract,runtime,opaque,inventory,approved_mounts):
    """The caller binds the measured raw freeze only after comparing the full normalized lock."""
    args=frozen_command(session,stage,work,destination,root,contract)
    script=Path(args[3])
    source=script.read_text()
    expected_freeze=contract.p["environment"]
    old_freeze="8bb71679bf5735f6e3c8225885be88c1f530639094df310a5a29e6691498455b"
    if stage=="decision":
        source=once(source,old_freeze,expected_freeze)
    else:
        tree=ast.parse(source)
        injection,=[n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                    and isinstance(n.value.func,ast.Name) and n.value.func.id=="exec"]
        probe=injection.value.args[0].value
        start=probe.index('    kaggle = Path("/kaggle/input")')
        end=probe.index('    source=base/',start)
        replacement=('    from src.runner_5a5.kaggle import inputs\n'
                     f'    approved=inputs(Path("/kaggle/input"),{approved_mounts!r})\n'
                     '    record["approved_inputs"]={name:str(path) for name,path in approved.items()}\n'
                     '    if any(v["accessible"] for k,v in reads.items() if k!="/kaggle/input"):\n'
                     '        raise ValueError("Administrative path accessible")\n')
        probe=probe[:start]+replacement+probe[end:]
        probe=probe.replace('/content/ns64_clean',str(runtime)).replace('Path("/content"),Path("/root")',
             'Path("/kaggle/working"),Path("/root"),Path("/content")')
        new_call=f"sys.path.insert(0,{str(session.KIT)!r})\nexec({probe!r},{{'__file__':__file__}})"
        source=once(source,ast.get_source_segment(source,injection),new_call)
        source=source.replace("Path('/content/ns64_clean')",f"Path({str(runtime)!r})")
        source=once(source,"Path('/content/imc2024_dioscuri_baalshamin_opaque.zip')) as archive",
                    f"Path({str(opaque)!r})) as archive")
        source=source.replace("Path('/content/imc2024_dioscuri_baalshamin_opaque.zip')",f"Path({str(opaque)!r})")
        source=once(source,old_freeze+"'",expected_freeze+"'") if old_freeze!=expected_freeze else source
        source=once(source,"environment['gpu']=='Tesla T4, 580.82.07, 15360 MiB'",
                    f"environment['gpu']=={inventory!r}")
        source=once(source,"torch.cuda.get_device_name(0)=='Tesla T4'",
                    "torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4'")
    compile(source,str(script),"exec")
    script.write_text(source)
    return [str(runtime/"venv/bin/python"),*args[1:]]
