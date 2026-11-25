from loguuuuuuu import logger


class Config:
    input_dir_root = "/kaggle/input"
    output_dir = "/tmp"
    check_exist_dir = False

    input_csv = "/kaggle/input/image-matching-challenge-2025/sample_submission.csv"
    category_csv = "/kaggle/input/image-matching-challenge-2024/test/categories.csv"
    target_datasets = []

    pipeline_json = "/kaggle/input/imc25-codes/configs/rotate_640_840_1024_1280_crop.json"
    # transparent_pipeline_json = "/kaggle/input/imc2024-pipelines/lg1280+1536+1024+2048_crop-lg1024+1280+1536/transp_pipeline.json"

    colmap_mapper_options = {
        "min_model_size": 3,  # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 5,
        # "num_threads": 1,
    }
