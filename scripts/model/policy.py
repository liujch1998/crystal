from typing import Union, List, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from model.t5 import T5ForConditionalGenerationAndTokenRegression
from utils.utils import logits_to_entropy, mask_pad, append_string


class Policy:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 policy_reward_sharing: bool,
                 accelerator,
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.policy_reward_sharing = policy_reward_sharing
        self.accelerator = accelerator

        # if policy_value_sharing:
        #     self.model = T5ForConditionalGenerationAndTokenRegression.from_pretrained(model_type)
        # else:
        self.model = T5ForConditionalGeneration.from_pretrained(model_type)
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)
        if model_ckpt is not None:
            checkpoint = torch.load(model_ckpt, map_location='cpu')
            self.model.load_state_dict(checkpoint, strict=False)
            checkpoint.clear()
        self.model.eval()

    def sample(self,
               questions_input_ids: torch.Tensor, # (B, QL)
               questions_attention_mask: torch.Tensor, # (B, QL)
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        prompts_input_ids, prompts_attention_mask = append_string(questions_input_ids, questions_attention_mask, ' \\n Knowledge: ', self.tokenizer)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        knowledges_input_ids = unwrapped_model.generate(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            max_length=self.tokenizer.max_knowledge_len + 1,
            min_length=3,
            do_sample=sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            synced_gpus=True,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        knowledges_input_ids = knowledges_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])
        knowledges_input_ids = F.pad(knowledges_input_ids, (0, self.tokenizer.max_knowledge_len - knowledges_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, KL)
        knowledges_attention_mask = (knowledges_input_ids != self.tokenizer.pad_token_id).long()
        knowledges_text = self.tokenizer.batch_decode(knowledges_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        lowercased_knowledges = [knowledge.lower() for knowledge in knowledges_text]
        lowercased_knowledges_tok = self.tokenizer.batch_encode_plus(
            lowercased_knowledges,
            return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_knowledge_len).to(knowledges_input_ids.device)
        lowercased_knowledges_input_ids = lowercased_knowledges_tok.input_ids
        lowercased_knowledges_attention_mask = lowercased_knowledges_tok.attention_mask

        return {
            'knowledges_text': knowledges_text,
            'knowledges_input_ids': knowledges_input_ids, # (B, KL)
            'knowledges_attention_mask': knowledges_attention_mask, # (B, KL)
            'lowercased_knowledges_text': lowercased_knowledges,
            'lowercased_knowledges_input_ids': lowercased_knowledges_input_ids, # (B, KL)
            'lowercased_knowledges_attention_mask': lowercased_knowledges_attention_mask, # (B, KL)
        }

    def forward_pass(self,
                     questions_input_ids: torch.Tensor, # (B, QL)
                     questions_attention_mask: torch.Tensor, # (B, QL)
                     knowledges_input_ids: torch.Tensor, # (B, KL)
                     knowledges_attention_mask: torch.Tensor, # (B, KL)
                    ):
        prompts_input_ids, prompts_attention_mask = append_string(questions_input_ids, questions_attention_mask, ' \\n Knowledge: ', self.tokenizer)
        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(knowledges_input_ids, knowledges_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )
        knowledges_logits = outputs.logits # (B, KL, V)
        logprobs = F.log_softmax(knowledges_logits, dim=-1) # (B, KL, V)
        knowledges_logprobs = torch.gather(logprobs, 2, knowledges_input_ids[:, :, None]).squeeze(2) # (B, KL)
        knowledges_entropy = logits_to_entropy(knowledges_logits) # (B, KL)

        results = {
            'knowledges_logits': knowledges_logits, # (B, KL, V)
            'knowledges_logprobs': mask_pad(knowledges_logprobs, knowledges_attention_mask), # (B, KL)
            'knowledges_entropy': mask_pad(knowledges_entropy, knowledges_attention_mask), # (B, KL)
        }

        if self.policy_value_sharing:
            logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, KL)
            results.update({
                'knowledges_value': mask_pad(logits, knowledges_attention_mask, 0), # (B, KL)
            })

        return results
