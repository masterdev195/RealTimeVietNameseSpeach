import torchaudio
import inspect

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda _: None
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

try:
    import huggingface_hub as hf

    if not getattr(hf.hf_hub_download, "_patched_use_auth_token", False):
        signature = inspect.signature(hf.hf_hub_download)
        if "use_auth_token" not in signature.parameters:
            has_token_param = "token" in signature.parameters
            original_hf_hub_download = hf.hf_hub_download

            def patched_hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
                if token is None and use_auth_token is not None:
                    token = use_auth_token
                if has_token_param:
                    return original_hf_hub_download(*args, token=token, **kwargs)
                return original_hf_hub_download(*args, **kwargs)

            patched_hf_hub_download._patched_use_auth_token = True
            hf.hf_hub_download = patched_hf_hub_download
except Exception:
    pass
