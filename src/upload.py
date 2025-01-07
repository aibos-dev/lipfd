from huggingface_hub import HfApi, HfFolder

# 登录 Hugging Face
HfFolder.save_token("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")

# 上传文件
api = HfApi()
api.upload_file(
    path_or_fileobj="/home/yanyanhao00/projects/LipFD/LipFD/model_epoch_10.pth",
    path_in_repo="model_epoch_10.pth",
    repo_id="yh007/LipFD",
    repo_type="model",
)
