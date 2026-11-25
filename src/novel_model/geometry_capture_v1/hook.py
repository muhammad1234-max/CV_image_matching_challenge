"""Instrumentation installed only inside the approved Linux Haiper worker."""

from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import numpy as np
from PIL import Image
from .database import (
    OFFSET,
    file_sha,
    inventory,
    logical,
    read_only,
    save,
    sha,
    snapshot,
)
from .analysis import convert


def replay_copy(parent, dest, manifest, receipt):
    if receipt["method"] != "pass_through_v1" or any(
        k is not True for k in receipt["keep"]
    ):
        raise ValueError("only an identity mask can reach replay")
    if dest.exists():
        raise FileExistsError(dest)
    if file_sha(parent) != manifest["backup_sha256"]:
        raise ValueError("parent changed")
    shutil.copyfile(parent, dest)
    if file_sha(dest) != manifest["backup_sha256"]:
        raise ValueError("copy differs")
    with closing(read_only(dest)) as db:
        records = logical(db)
    if records != manifest["original_logical"]:
        raise ValueError("mapper rows changed")
    return records


def install(stack, impl, pipeline, work, source, config):
    import colmap
    from ..geometry_graph_v1.schemas import policy
    from ..geometry_graph_v1.graph_import import load_cache
    from ..geometry_graph_v1.diagnostics import checked_keep_mask

    out = work / "capture_v1"
    out.mkdir()
    params = pipeline.conf
    if (
        params.clustering
        or params.pruning
        or params.post_localizer
        or params.verification.type != "colmap"
    ):
        raise ValueError("unexpected verification or localization branch")
    offsets = {}
    verified = {}
    state = {}
    original_add = colmap.COLMAPDatabase.add_keypoints

    def add(db, image_id, points):
        xy = np.asarray(points[:, :2], dtype=np.float32).copy()
        xy += np.float32(0.5)
        offsets[image_id] = sha(xy.tobytes())
        return original_add(db, image_id, points)

    stack.enter_context(patch.object(colmap.COLMAPDatabase, "add_keypoints", add))
    original_verify = impl.verify_matches

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        for i, group in result.inliers.items():
            for j, idx in group.items():
                if len(idx):
                    verified[str(i), str(j)] = (
                        len(idx),
                        sha(np.ascontiguousarray(idx, dtype="<u4").tobytes()),
                    )
        return result

    stack.enter_context(patch.object(impl, "verify_matches", verify))
    original_map = impl.pycolmap.incremental_mapping
    original_export = impl.get_reconstructions

    def stage(name):
        save(out / "stage.json", dict(stage=name, monotonic=time.monotonic()))
        if time.monotonic() >= float(os.environ["STAIRS_DEADLINE"]) - 90:
            raise TimeoutError("reserved export time")

    def mapper(**kwargs):
        if state:
            raise ValueError("multiple mapper invocations")
        stage("snapshot")
        dbpath = Path(kwargs["database_path"])
        tmp = out / "parent.db"
        snap = snapshot(dbpath, tmp)
        parent = out / (snap["backup_sha256"] + ".db")
        tmp.rename(parent)
        names = [f"image_{i:04}.png" for i in range(54)]
        images, cameras, keys, pairs, arrays = inventory(parent, names, offsets)
        observed = {}
        for (i, j), pair in pairs.items():
            if "verified" in pair and len(pair["verified"]):
                observed[images[i]["name"], images[j]["name"]] = (
                    len(pair["verified"]),
                    sha(pair["verified"].tobytes()),
                )
        if observed != verified:
            raise ValueError("in-memory verifier and mapper input disagree")
        root = Path(kwargs["image_path"])
        image_hashes = {n: file_sha(root / n) for n in names}
        qual = {}
        for im in images.values():
            cam = cameras[im["camera_id"]]
            name = im["name"]
            with Image.open(root / name) as photo:
                if photo.size != (cam["width"], cam["height"]):
                    raise ValueError("camera dimensions mismatch")
                equiv = photo.getexif().get(41989)
                qual[name] = bool(
                    equiv
                    and np.isclose(
                        float(equiv) / 35 * max(photo.size), cam["params"][0], rtol=1e-5
                    )
                )
        identity = json.loads((work / "identity.json").read_text())
        env = json.loads((work / "environment.json").read_text())
        if (
            sha("\n".join(image_hashes.values()).encode())
            != identity["permutation_sha256"]
        ):
            raise ValueError("mapper image bytes or order changed")
        options = kwargs["options"].summary()
        manifest = dict(
            schema="ns64_pre_mapper_v1",
            stage="complete_pre_mapper_capture",
            **snap,
            source_revision=identity["source_revision"],
            source_inventory_hash=env["source_inventory_hash"],
            config_sha256=file_sha(config),
            image_sha256=image_hashes,
            seed=20261111,
            permutation_sha256=identity["permutation_sha256"],
            checkpoints=env["assets"],
            environment_receipt_sha256=file_sha(work / "environment.json"),
            worker_sha256=file_sha(work / "worker.py"),
            adapter_sha256=file_sha(Path(__file__)),
            adapter_bundle_sha256=os.environ["CAPTURE_BUNDLE_SHA256"],
            prototype_proposal_sha256=policy()["proposal_sha256"],
            offset_sha256=offsets,
            coordinates=OFFSET,
            pair_convention="COLMAP ascending database IDs, first to second",
            camera_parameters={
                "SIMPLE_PINHOLE": "f,cx,cy",
                "PINHOLE": "fx,fy,cx,cy",
                "SIMPLE_RADIAL": "f,cx,cy,k",
                "RADIAL": "f,cx,cy,k1,k2",
                "OPENCV": "fx,fy,cx,cy,k1,k2,p1,p2",
            },
            mapper_options=options,
            verification=params.verification.model_dump(mode="json"),
            images=images,
            cameras=cameras,
            focal_quality=qual,
            arrays=arrays,
            count=dict(
                images=len(images),
                raw_pairs=sum("raw" in p for p in pairs.values()),
                verified_pairs=len(observed),
                keypoints=sum(map(len, keys.values())),
                verified_indices=sum(n for n, h in observed.values()),
            ),
        )
        save(out / "manifest.json", manifest)
        stage("diagnostics")
        convert(parent, manifest, out)
        state.update(parent=parent, manifest=manifest, kwargs=kwargs)
        return run_map("original")

    def run_map(branch):
        stage(branch + "_mapper")
        m = state["manifest"]
        parent = state["parent"]
        target = work / (branch + ".db")
        if branch == "original":
            shutil.copyfile(parent, target)
            if file_sha(target) != m["backup_sha256"]:
                raise ValueError("original parent copy differs")
            with closing(read_only(target)) as db:
                before = logical(db)
        else:
            mask = json.loads((out / "pass_through_v1.json").read_bytes())
            graph = load_cache(
                out / "graph.json", json.loads((out / "graph_trust.json").read_bytes())
            )
            checked_keep_mask(mask, graph, policy())
            before = replay_copy(parent, target, m, mask)
        if before != m["original_logical"]:
            raise ValueError("input record parity failed")
        kwargs = state["kwargs"].copy()
        kwargs["database_path"] = str(target)
        if branch != "original":
            destination = work / "pass_through_maps"
            destination.mkdir()
            kwargs["output_path"] = str(destination)
        if kwargs["options"].summary() != m["mapper_options"]:
            raise ValueError("mapper options changed")
        save(
            out / (branch + "_input.json"),
            dict(
                database_sha256=file_sha(target),
                records=before,
                options=m["mapper_options"],
                permutation_sha256=m["permutation_sha256"],
                environment_receipt_sha256=m["environment_receipt_sha256"],
            ),
        )
        start = time.monotonic()
        maps = original_map(**kwargs)
        save(
            out / (branch + "_map.json"),
            dict(seconds=time.monotonic() - start, models=len(maps)),
        )
        return maps

    def export(maps, scene, *args, **kwargs):
        def emit(branch, mapping):
            result = original_export(mapping, scene, *args, **kwargs)
            frame = impl.results_to_submission_df(
                {scene.dataset: {scene.scene: result[0]}}, schema="imc2025"
            )
            if sorted(frame.image) != sorted(state["manifest"]["image_sha256"]):
                raise ValueError("mapper export coverage")
            frame.to_csv(out / (branch + ".csv"), index=False)
            return result

        original = emit("original", maps)
        emit("pass_through", run_map("pass_through"))
        stage("both_exports_complete")
        return original

    stack.enter_context(patch.object(impl.pycolmap, "incremental_mapping", mapper))
    stack.enter_context(patch.object(impl, "get_reconstructions", export))
