from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import List, Optional, Tuple, Union

from transformers.utils.output_capturing import OutputRecorder
from transformers.utils import auto_docstring, logging
from transformers.utils.generic import  TransformersKwargs
from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (Qwen3OmniMoePreTrainedModelForConditionalGeneration, Qwen3OmniMoeAudioEncoder,
                                                                        Qwen3OmniMoeThinkerTextDecoderLayer,
                                                                        Qwen3OmniMoeThinkerTextAttention, Qwen3OmniMoeThinkerTextSparseMoeBlock)
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput

from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DynamicCache, Qwen3_5TextModel,
                                                          Qwen3_5VisionModel)

from .configuration_tlive_omni import TLiveOmniConfig, TLiveOmniAudioEncoderConfig

from .processing_tlive_omni import _get_feat_extract_output_lengths


logger = logging.get_logger(__name__)


def _require_flash_attention_3d_mrope_support():
    from transformers.modeling_flash_attention_utils import _is_packed_sequence

    remediation = (
        "TLive-Omni FlashAttention2 requires the supported Transformers 5.2.0 wheel "
        "with 3D M-RoPE support. Install the wheel documented by this release, then restart Python."
    )
    probe = torch.arange(4, dtype=torch.long).view(1, 1, 4).expand(3, 1, 4)
    try:
        is_packed = _is_packed_sequence(probe, batch_size=1)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(remediation) from exc
    if bool(is_packed):
        raise RuntimeError(remediation)


class TLiveOmniAudioEncoder(Qwen3OmniMoeAudioEncoder):
    config: TLiveOmniAudioEncoderConfig

    def __init__(self, config: TLiveOmniAudioEncoderConfig):
        super().__init__(config)

 

    # Keep convolution collectives aligned when audio lengths differ across ZeRO-3 ranks.
    def _chunked_conv_forward(self, padded_feature):
        original_len = padded_feature.size(0)
        
        # Round up to the number of local convolution chunks.
        local_chunk_num = (original_len + self.conv_chunksize - 1) // self.conv_chunksize
        
        # Synchronization is only needed across multiple distributed ranks.
        is_distributed = (
            dist.is_available() and 
            dist.is_initialized() and 
            dist.get_world_size() > 1
        )
        
        if not is_distributed:
            # A single rank can process its local chunks without synchronization.
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embed = F.gelu(self.conv2d1(chunk))
                padded_embed = F.gelu(self.conv2d2(padded_embed))
                padded_embed = F.gelu(self.conv2d3(padded_embed))
                padded_embeds.append(padded_embed)
            
            if len(padded_embeds) == 1:
                return padded_embeds[0]
            else:
                return torch.cat(padded_embeds, dim=0)
        
        # Compare the minimum and maximum chunk counts across ranks.
        min_chunk_tensor = torch.tensor(local_chunk_num, device=padded_feature.device, dtype=torch.long)
        max_chunk_tensor = torch.tensor(local_chunk_num, device=padded_feature.device, dtype=torch.long)
        
        dist.all_reduce(min_chunk_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_chunk_tensor, op=dist.ReduceOp.MAX)
        
        global_min_chunk = min_chunk_tensor.item()
        global_max_chunk = max_chunk_tensor.item()
        
        if global_min_chunk == global_max_chunk:
            # Equal chunk counts need no cross-rank padding.
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                padded_embed = F.gelu(self.conv2d1(chunk))
                padded_embed = F.gelu(self.conv2d2(padded_embed))
                padded_embed = F.gelu(self.conv2d3(padded_embed))
                padded_embeds.append(padded_embed)
            
            if len(padded_embeds) == 1:
                return padded_embeds[0]
            else:
                return torch.cat(padded_embeds, dim=0)
        
        # Otherwise, synchronize to the largest feature length.
        global_max_len_tensor = torch.tensor(original_len, device=padded_feature.device)
        dist.all_reduce(global_max_len_tensor, op=dist.ReduceOp.MAX)
        global_max_len = global_max_len_tensor.item()
        
        # Pad along the batch dimension.
        if original_len < global_max_len:
            pad_size = global_max_len - original_len
            padded_feature_padded = F.pad(padded_feature, (0, 0, 0, 0, 0, 0, 0, pad_size))
            logger.debug(
                "[RANK %s] AUDIO padded from %s to %s",
                dist.get_rank(),
                original_len,
                global_max_len,
            )
        else:
            padded_feature_padded = padded_feature
        
        # Process the globally padded input in aligned chunks.
        padded_embeds = []
        for chunk in padded_feature_padded.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        
        # Merge the aligned chunks.
        if len(padded_embeds) == 1:
            final_embed = padded_embeds[0]
        else:
            final_embed = torch.cat(padded_embeds, dim=0)
        
        # Restore the original local length.
        final_embed = final_embed[:original_len]
        
        return final_embed

    @auto_docstring
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embed = self._chunked_conv_forward(padded_feature)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for layer_num, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]


        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class TLiveOmniTextModel(Qwen3_5TextModel):

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and not isinstance(past_key_values, Qwen3_5DynamicCache):
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # mrope: the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)


        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    def _update_linear_attn_mask(self, attention_mask, cache_position):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

@dataclass
class TLiveOmniCausalLMOutputWithPast(ModelOutput):
    """
    Args:
        logits: ...
        past_key_values: ...
        ...
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    token_accuracy: Optional[torch.FloatTensor] = None

def find_audio_end_token_indice(tensor, A):
    candidates = tensor[tensor > A]
    if len(candidates) == 0:
        raise ValueError(f"Could not find an audio end token index after {A} in {tensor}.")
    return torch.min(candidates).item()


class TLiveOmniForConditionalGeneration(Qwen3OmniMoePreTrainedModelForConditionalGeneration, GenerationMixin):
    config_class = TLiveOmniConfig
    config: TLiveOmniConfig
    accepts_loss_kwargs = False
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _no_split_modules = [
        "Qwen3OmniMoeAudioEncoderLayer",
        "Qwen3OmniMoeThinkerTextDecoderLayer",
    ]
    _can_record_outputs = {
        "hidden_states": Qwen3OmniMoeThinkerTextDecoderLayer,
        "attentions": Qwen3OmniMoeThinkerTextAttention,
        "router_logits": OutputRecorder(Qwen3OmniMoeThinkerTextSparseMoeBlock, index=1),
    }

    def __init__(self, config):
        attention_implementations = {
            getattr(component, "_attn_implementation", None)
            for component in (config, config.audio_config, config.text_config, config.vision_config)
        }
        if "flash_attention_2" in attention_implementations:
            _require_flash_attention_3d_mrope_support()
        super().__init__(config)
        self.audio_tower = TLiveOmniAudioEncoder._from_config(config.audio_config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = TLiveOmniTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        self.post_init()


    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds

        return vision_output.pooler_output

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: Optional[torch.LongTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            feature_attention_mask (`torch.LongTensor`, *optional*):
                Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """
        input_features = input_features.to(dtype=self.audio_tower.dtype)
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_features = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )

        return audio_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device))
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}")

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}")

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    @auto_docstring
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> Union[tuple, TLiveOmniCausalLMOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        use_audio_in_video (`bool`, *optional*):
            Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
        video_second_per_grid (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            Number of seconds per grid for each video, used for temporal feature mapping.

        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoProcessor
        >>> model_id = "TaoLiveAIGC/TLive-Omni-4B"
        >>> processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        >>> model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        >>> model_inputs = processor(text="Describe the input.", return_tensors="pt")
        >>> generated_ids = model.generate(**model_inputs, max_new_tokens=128)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)
        ```"""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)

            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)


        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)



        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)

        return TLiveOmniCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        model_inputs["position_ids"] = None
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["input_features"] = None

        return model_inputs
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        audio_start_token_id = self.config.audio_start_token_id
        audio_end_token_id = self.config.audio_end_token_id
        position_id_per_seconds = self.config.position_id_per_seconds

        # Timestamps split videos into per-temporal-grid spans in the
        # audio-in-video path.
        video_chunk_end_indices = set()
        if video_grid_thw is not None:
            if second_per_grids is None:
                second_per_grids = torch.ones(
                    video_grid_thw.shape[0],
                    dtype=torch.float,
                )
            else:
                second_per_grids = torch.as_tensor(second_per_grids, dtype=torch.float).detach().cpu()
            second_per_grids = torch.repeat_interleave(
                second_per_grids, video_grid_thw[:, 0].detach().cpu(), dim=0
            )
            video_chunk_end_indices = set(torch.cumsum(video_grid_thw[:, 0], dim=0).tolist())
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
            
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
                image_nums, video_nums, audio_nums = 0, 0, 0
                # Audio-in-video spans use
                # <|vision_start|><|audio_start|>video_pad* audio_pad*<|audio_end|><|vision_end|>.
                vision_end_indices = torch.argwhere(input_ids == vision_end_token_id).squeeze(1)
                audio_end_indices = torch.argwhere(input_ids == audio_end_token_id).squeeze(1)
                vision_tokens = input_ids[vision_end_indices - 1]
                audio_start_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_audio_nums = (vision_tokens == audio_end_token_id).sum()
                video_only_nums = (vision_tokens == video_token_id).sum()
                video_nums = video_audio_nums if use_audio_in_video and video_audio_nums > 0 else video_only_nums
                audio_nums = audio_start_nums - video_audio_nums
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = image_nums + video_nums + audio_nums
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                        remain_videos > 0 or remain_images > 0
                    ):
                        ed_vision_start = input_tokens.index(vision_start_token_id, st)
                        ed_vision_end = input_tokens.index(vision_end_token_id, st)
                    else:
                        ed_vision_start = len(input_tokens) + 1
                    try:
                        next_audio_start = input_tokens.index(audio_start_token_id, st)
                    except ValueError:
                        next_audio_start = len(input_tokens) + 1
                    if use_audio_in_video and ed_vision_start < next_audio_start < ed_vision_end:
                        ed_audio_start = next_audio_start
                    elif remain_audios > 0:
                        ed_audio_start = next_audio_start
                    else:
                        ed_audio_start = len(input_tokens) + 1
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                        st_idx += text_len
                    bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif min_ed == ed_vision_start and input_ids[ed_vision_start + 1] == image_token_id:
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).float()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif min_ed == ed_vision_start and ed_vision_end < ed_audio_start:

                        grid_t = torch.tensor(1, device=input_ids.device, dtype=video_grid_thw.dtype)
                        grid_hs = video_grid_thw[video_idx][1]
                        grid_ws = video_grid_thw[video_idx][2]
                        
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            grid_t.item(),
                            grid_hs.item() // spatial_merge_size,
                            grid_ws.item() // spatial_merge_size,
                        )
                        
                        t_index = (
                            torch.arange(llm_grid_t, dtype=torch.float)
                            * second_per_grids[video_idx]
                            * position_id_per_seconds
                        ).add(1e-6).long()
                        t_index = t_index.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
                        video_len = grid_t * grid_hs * grid_ws // (spatial_merge_size**2)
                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif min_ed == ed_vision_start and ed_vision_end > ed_audio_start:
                        eos_len = 2
                        if ed_audio_start != ed_vision_start + bos_len:
                            raise ValueError(
                                "Audio-in-video span expected audio_start immediately after vision_start."
                            )
                        grid_hs = video_grid_thw[video_idx][1]
                        grid_ws = video_grid_thw[video_idx][2]
                        grid_tokens = grid_hs * grid_ws // (spatial_merge_size**2)
                        audio_end_token_indice = find_audio_end_token_indice(audio_end_indices, ed_audio_start)
                        if audio_end_token_indice != ed_vision_end - 1:
                            raise ValueError(
                                "Audio-in-video span expected audio_end immediately before vision_end."
                            )
                        video_token_start = ed_audio_start + 1
                        video_token_end = video_token_start
                        while video_token_end < audio_end_token_indice and input_ids[video_token_end] == video_token_id:
                            video_token_end += 1
                        actual_video_len = video_token_end - video_token_start
                        if actual_video_len % grid_tokens != 0:
                            raise ValueError(
                                f"Video/audio span has {actual_video_len} video tokens, "
                                f"which is not divisible by one temporal grid of {grid_tokens} tokens."
                            )
                        grid_t = actual_video_len // grid_tokens
                        if grid_t != 1:
                            raise ValueError(
                                f"Audio-in-video span expected exactly one temporal grid, got {grid_t}."
                            )
                        
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            grid_t.item(),
                            grid_hs.item() // spatial_merge_size,
                            grid_ws.item() // spatial_merge_size,
                        )
                        
                        t_index = (
                            torch.arange(llm_grid_t, dtype=torch.float)
                            * second_per_grids[video_idx]
                            * position_id_per_seconds
                        ).add(1e-6).long()
                        audio_start_st_idx = st_idx
                        content_st_idx = st_idx + 1
                        t_index = t_index.view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        audio_start_llm_pos_ids = torch.arange(1).view(1, -1).expand(3, -1) + audio_start_st_idx
                        llm_pos_ids_list.append(audio_start_llm_pos_ids)
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + content_st_idx)
                        video_len = grid_t * grid_hs * grid_ws // (spatial_merge_size**2)

                        audio_len = audio_end_token_indice - video_token_end
                        audio_len_int = audio_len.item() if torch.is_tensor(audio_len) else audio_len
                        if audio_len_int > 0 and not torch.all(
                            input_ids[video_token_end:audio_end_token_indice] == audio_token_id
                        ):
                            raise ValueError(
                                "Audio-in-video span expected only audio_pad tokens between video_pad and audio_end."
                            )
                        if audio_len_int > 0:
                            audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + content_st_idx
                            llm_pos_ids_list.append(audio_llm_pos_ids)
                        st += int(text_len + bos_len + 1 + audio_len + video_len + eos_len)

                        video_idx += int(grid_t.item())
                        if video_idx in video_chunk_end_indices:
                            audio_idx += 1
                        remain_videos -= 1
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1).long()

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas
