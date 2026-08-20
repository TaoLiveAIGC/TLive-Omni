from packaging.version import Version
from transformers import __version__ as transformers_version
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig

SUPPORTED_TRANSFORMERS_VERSION = "5.2.0"
if Version(transformers_version) != Version(SUPPORTED_TRANSFORMERS_VERSION):
    raise ImportError(
        "TLive-Omni Hub custom code requires "
        f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}, but found {transformers_version}. "
        "Install the TLive Transformers wheel documented by this release."
    )

class TLiveOmniAudioEncoderConfig(PretrainedConfig):

    model_type = "tlive_omni_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        deepstack_audio_indexes=[8, 16, 24],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size
        self.deepstack_audio_indexes = deepstack_audio_indexes


class TLiveOmniConfig(PretrainedConfig):

    model_type = "tlive_omni"
    # Override parent's attribute_map as we use audio_token_id directly, not audio_token_index
    attribute_map = {}
    sub_configs = {
        "audio_config": TLiveOmniAudioEncoderConfig,
        "vision_config": Qwen3_5VisionConfig,
        "text_config": Qwen3_5TextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id=248076,
        image_token_id=248056,
        video_token_id=248057,
        position_id_per_seconds=13,
        audio_start_token_id=248070,
        audio_end_token_id=248071,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.initializer_range = initializer_range

        if isinstance(vision_config, dict):
            vision_config = Qwen3_5VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3_5VisionConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = TLiveOmniAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = TLiveOmniAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_5TextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
