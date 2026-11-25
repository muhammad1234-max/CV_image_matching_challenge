# Active checkpoint inventory

Prior Kaggle PASS for source and images is independent of model loading. All seven GPU loader checks are pending. Meta publishes the DINOv2 SHA256 as the LFS object identity, matching the Kaggle mount. Other rows have no confirmed author-published SHA256. Hashes below are project measurements, from the original download or the previous pinned Kaggle mount. An unavailable value is explicitly marked pending. Sizes are bytes.

| Model | Expected file and authoritative source | Size | SHA256 | Previous mount |
|---|---|---:|---|---|
| superpoint | [superpoint_v1.pth](https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/ddcf11f42e7e0732a0c4607648f9448ea8d73590/models/weights/superpoint_v1.pth) | 5206086 | 52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e | missing |
| aliked | [aliked-n16.pth](https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth) | 2738091 | 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2 | present |
| mast3r | [MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth) | 2754910614 | e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2 | missing |
| retrieval | [MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth) | 8399111 | 000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50 | missing |
| codebook | [MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl](https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl) | 268435691 | 700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a | missing |
| isc | [isc_ft_v107.pth.tar](https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar) | 420418996 | 512390a15ab52f8ab5adc14a315b157a5d6e732e3361c7a0dce5be6e95a86d30 | missing |
| dinov2 | [pytorch_model.bin](https://huggingface.co/facebook/dinov2-base/blob/main/pytorch_model.bin) | 346392713 | 014965d9e330e7f4bff8ddcbee9df5e4f2ca032b2f5180942a6edb454783e75d | present |

| Model | Architecture and serialization | Required module | License context |
|---|---|---|---|
| superpoint | SuperPoint CNN, state dict | models._research_only.magicleap.superpoint | Magic Leap academic or nonprofit noncommercial research license |
| aliked | ALIKED N16 state dict | features.lightglue.aliked | ALIKED BSD-3-Clause; retain upstream notices |
| mast3r | AsymmetricMASt3R ViT-Large / base decoder 512 PyTorch checkpoint | models.mast3r.model | CC BY-NC-SA 4.0 and upstream training dataset conditions, including MapFree. Review CHECKPOINTS_NOTICE. |
| retrieval | training-free retrieval head checkpoint | mast3r.retrieval.processor | CC BY-NC-SA 4.0 and upstream training dataset conditions, including MapFree. Review CHECKPOINTS_NOTICE. |
| codebook | 64k ASMK codebook pickle | asmk.asmk_method | CC BY-NC-SA 4.0 and upstream training dataset conditions, including MapFree. Review CHECKPOINTS_NOTICE. |
| isc | ISCNet EfficientNetV2-M fine-tuned v107 PyTorch tar checkpoint | models.isc.model | ISC repository MIT; training-data rights and checkpoint redistribution permission require separate review |
| dinov2 | ViT-B/14 Transformers directory | global_descriptors.dinov2 | Meta Apache-2.0 for code and weights |

The retrieval loader derives its codebook filename from the head filename, so those two must be colocated. DINOv2 also needs both hashed JSON sidecars. The alternative Hugging Face MASt3R safetensors file has a different serialization. DAD and VGGT have no active weight references in the selected YAML.

[ns64 model paths](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/07b5695b99a8ae4725d5d952f0065c2157fadd7d/conf/models.yaml) | [NAVER checkpoint notice](https://github.com/naver/mast3r/blob/main/CHECKPOINTS_NOTICE) | [Magic Leap terms](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE) | [ALIKED](https://github.com/Shiaoming/ALIKED) | [ISC](https://github.com/lyakaap/ISC21-Descriptor-Track-1st)
