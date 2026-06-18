
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


class GatedConceptStream(nn.Module):
    """

    main architecture 
   

    ĥ_t  = LayerNorm(h_t)
    r_t  = σ(W_r · ĥ_t)
    f_t  = σ(W_f · ĥ_t)
    w_t  = σ(W_w · ĥ_t)
    h'_t = (1 - f_t) ⊙ h_t  +  r_t ⊙ c_{t-1}
    c_t  = LayerNorm(c_{t-1} + w_t ⊙ h'_t)
    """

    def __init__(self, d_model):
        super().__init__()

        self.ln_h        = nn.LayerNorm(d_model)
        self.ln_residual = nn.LayerNorm(d_model)

        self.read_gate   = nn.Linear(d_model, d_model)
        self.forget_gate = nn.Linear(d_model, d_model)
        self.write_gate  = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.read_gate.weight)
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.zeros_(self.write_gate.weight)

      
        self.read_gate.bias.data.fill_(-0.28)   
        self.forget_gate.bias.data.fill_(-1.0)  
        self.write_gate.bias.data.fill_(-1.5)   

        self.last_r = 0.43
        self.last_f = 0.27
        self.last_w = 0.18

    def forward(self, h_t, concept_stream_prev):
        """
        Args:
            h_t:                 [1, d_model] current hidden state
            concept_stream_prev: [1, d_model] residual from previous pass
        Returns:
            h_gated:             [1, d_model]
            concept_stream_new:  [1, d_model]
        """
        h_norm = self.ln_h(h_t)

        r = torch.sigmoid(self.read_gate(h_norm))
        f = torch.sigmoid(self.forget_gate(h_norm))
        w = torch.sigmoid(self.write_gate(h_norm))

        with torch.no_grad():
            self.last_r = r.mean().item()
            self.last_f = f.mean().item()
            self.last_w = w.mean().item()

        h_gated            = (1 - f) * h_t + r * concept_stream_prev
        concept_stream_new = self.ln_residual(concept_stream_prev + w * h_gated)

        return h_gated, concept_stream_new


class AGCLR(nn.Module):
  

    def __init__(self, base_causallm, latent_token_id, start_latent_id,
                 end_latent_id, eos_token_id):
        super(AGCLR, self).__init__()

        self.gen_forward_cnt = 0
        self.base_causallm   = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id    = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id   = end_latent_id

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
            self.d_model   = self.base_causallm.transformer.wte.embedding_dim
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            self.d_model   = (
                self.base_causallm.config.hidden_size
                if hasattr(self.base_causallm.config, 'hidden_size')
                else self.base_causallm.config.n_embd
            )

        self.gated_concept_stream = GatedConceptStream(self.d_model)

        total_params = sum(p.numel() for p in self.parameters())
        gate_params  = sum(p.numel() for p in
                           self.gated_concept_stream.parameters())

        print(f" AGCLR initialized")
        print(f"   d_model      : {self.d_model}")
        print(f"   Gate params  : {gate_params:,}  "
              f"({100 * gate_params / total_params:.2f}% of total)")
        print(f"   Total params : {total_params:,}")
        print(f"   Gate init    : R={self.gated_concept_stream.last_r:.2f}  "
              f"F={self.gated_concept_stream.last_f:.2f}  "
              f"W={self.gated_concept_stream.last_w:.2f}")

   

    def train(self, mode=True):
        super().train(mode)
        self.base_causallm.train(mode)
        return self

    def eval(self):
        return self.train(False)

  

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        logits = []

       
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists   = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]
        max_n_latents = max((len(l) for l in latent_lists), default=0)

        
        concept_streams = torch.zeros(batch_size, self.d_model, device=device)

        inputs_embeds      = self.embedding(input_ids)
        next_compute_range = (0, seq_len)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[
                        :, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :],
                     v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                seq_len if pass_idx + 1 >= max_n_latents
                        else next_compute_range[1] + 1,
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache      = outputs.past_key_values

            filling_indices = [
                (i, latent_lists[i][pass_idx])
                for i in range(batch_size)
                if len(latent_lists[i]) > pass_idx
            ]

            tensor_list = [
                [inputs_embeds[b, p, :]
                 for p in range(inputs_embeds.shape[1])]
                for b in range(batch_size)
            ]

            updated_concept_streams = concept_streams.clone()

            for b, token_idx in filling_indices:
                h_prev = hidden_states[
                    b, token_idx - 1 - hidden_states_offset, :]

                h_gated, c_new = self.gated_concept_stream(
                    h_prev.unsqueeze(0),
                    concept_streams[b].unsqueeze(0)
                )

                tensor_list[b][token_idx]  = h_gated.squeeze(0)
                updated_concept_streams[b] = c_new.squeeze(0)

            concept_streams = updated_concept_streams

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[b])
                for b in range(batch_size)
            ])

      
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[
                :, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [(k[:, :, :next_compute_range[0], :],
                  v[:, :, :next_compute_range[0], :])
                 for k, v in kv_cache]
                if kv_cache else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1

        logits       = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss         = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


    def generate(self, input_ids, attention_mask,
                 max_new_tokens=16, output_embedding=False,
                 synced_gpus=False, **kwargs):

        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only batch_size=1 supported"

        tokens = input_ids[0].detach().tolist()

        labels  = input_ids.clone()
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids),
            labels,
            torch.arange(input_ids.shape[1],
                         dtype=torch.long,
                         device=input_ids.device).unsqueeze(0),
        )
        inputs_embeds = outputs.inputs_embeds

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_emb = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_emb), dim=1)

        for _ in range(max_new_tokens - 1):
            out        = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_emb = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_emb), dim=1)

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        return (torch.tensor(tokens).view(1, -1), new_inputs_embeds) \
               if output_embedding else torch.tensor(tokens).view(1, -1)



def create_agclr_from_cot_checkpoint(checkpoint_path, model_id, tokenizer,
                                      device, start_id, end_id, latent_id):
    checkpoint = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=False)

    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(model_id)
    base.resize_token_embeddings(len(tokenizer))
    initialize_special_tokens(base, tokenizer, start_id, end_id, latent_id)
    base.load_state_dict(checkpoint['model_state_dict'])

    agclr = AGCLR(
        base_causallm    = base,
        latent_token_id  = latent_id,
        start_latent_id  = start_id,
        end_latent_id    = end_id,
        eos_token_id     = tokenizer.eos_token_id,
    ).to(device)

    print(f"   Loaded from CoT checkpoint — epoch {checkpoint['epoch']}, "
          f"loss {checkpoint.get('train_loss', float('nan')):.4f}")
    return agclr







    'AGCLR':                          AGCLR,
    'GatedConceptStream':             GatedConceptStream,
    'create_agclr_from_cot_checkpoint': create_agclr_from_cot_checkpoint,
    'test_agclr':                     test_agclr,
})
