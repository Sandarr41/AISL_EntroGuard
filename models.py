import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------------
# Модель эмбеддинга на стороне сервиса f : текст -> R^d (заморожена, чёрный
# ящик, как Sentence-T5 в статье -- мы её никогда не обучаем, это просто
# фиксированное отображение)
# ---------------------------------------------------------------------------
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, nlayers=2, max_len=16):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.out_proj = nn.Linear(d_model, d_model)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, token_ids, pad_id):
        pos = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        mask = token_ids.eq(pad_id)
        h = self.encoder(x, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        lengths = (~mask).sum(1, keepdim=True).clamp(min=1)
        pooled = h.sum(1) / lengths
        return self.out_proj(pooled)


# ---------------------------------------------------------------------------
# Обучаемый EIA на основе transformer decoder-only (в стиле GEIA).
# Эмбеддинг проецируется и вставляется как псевдо-токен-префикс; далее
# causal-трансформер авторегрессивно восстанавливает текст.
# ---------------------------------------------------------------------------
class TransformerAttacker(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model=64, nhead=4, nlayers=3, max_len=17):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.emb_proj = nn.Linear(emb_dim, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                        batch_first=True, activation="gelu")
            for _ in range(nlayers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tok_emb.weight  # связаны, как Wdec в статье

    def _causal_mask(self, n, device):
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, embedding, input_ids, return_block_outputs=False):
        """embedding: (B, d_emb); input_ids: (B, T) токены, подаваемые как вход
        декодера (с префиксом BOS). Возвращает logits (B, T, V) и, опционально,
        промежуточные выходы по блокам, нужные для entropy / logit-lens лосса."""
        B, T = input_ids.shape
        prefix = self.emb_proj(embedding).unsqueeze(1)          # (B,1,d)
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device)).unsqueeze(0)
        seq = torch.cat([prefix, tok + pos], dim=1)              # (B,1+T,d)
        mask = self._causal_mask(seq.size(1), seq.device)

        block_outputs = []
        h = seq
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
            block_outputs.append(h)

        logits = self.head(self.ln_f(h))[:, 1:, :]  # отбрасываем позицию префикса
        if return_block_outputs:
            return logits, block_outputs
        return logits

    def block_distributions(self, block_outputs):
        """Logits-lens: применяем *финальную* норму + связанную выходную
        голову к скрытому состоянию каждого промежуточного блока ->
        распределение слов на каждом блоке."""
        dists = []
        for h in block_outputs:
            d = F.softmax(self.head(self.ln_f(h))[:, 1:, :], dim=-1)  # (B,T,V)
            dists.append(d)
        return dists

    @torch.no_grad()
    def greedy_decode(self, embedding, bos_id, eos_id, max_len):
        B = embedding.size(0)
        device = embedding.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(embedding, generated)
            next_tok = logits[:, -1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(eos_id)
            if finished.all():
                break
        return generated


# ---------------------------------------------------------------------------
# Нетрансформерный EIA: многослойный LSTM-декодер. Архитектурно не связан
# со структурой transformer-блоков, против которой обучался/проектировался
# EntroGuard -- используется для проверки переноса за пределы семейства.
# ---------------------------------------------------------------------------
class LSTMAttacker(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model=64, nlayers=3, max_len=17):
        super().__init__()
        self.d_model = d_model
        self.nlayers = nlayers
        self.emb_proj = nn.Linear(emb_dim, d_model * nlayers)  # задаёт (h0,c0) для каждого слоя
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=nlayers, batch_first=True)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tok_emb.weight

    def _init_state(self, embedding):
        B = embedding.size(0)
        h0 = torch.tanh(self.emb_proj(embedding)).view(B, self.nlayers, self.d_model).transpose(0, 1).contiguous()
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, embedding, input_ids, return_layer_outputs=False):
        h0, c0 = self._init_state(embedding)
        tok = self.tok_emb(input_ids)
        # Backward для RNN в cuDNN поддерживается только для *train-режима*
        # ядер; обучение entroguard/adaptive-attacker вызывает это, когда
        # атакующий заморожен и переведён в .eval() (dropout здесь всё
        # равно 0, так что eval() vs train() численно ничего не меняет),
        # но backward через него всё равно нужен -- cuDNN на CUDA кидает
        # "backward can only be called in training mode" именно для этой
        # комбинации (у CPU-реализации такого ограничения нет, поэтому
        # раньше баг не проявлялся). Отключение cuDNN для этой крошечной
        # LSTM ничего не стоит по скорости и полностью снимает ограничение.
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.lstm(tok, (h0, c0))
        logits = self.head(self.ln_f(out))
        if return_layer_outputs:
            # скрытые состояния по каждому шагу и слою через ручной разворот
            # (аналог "снятия показаний с блоков transformer'а", но по
            # глубине рекуррентных слоёв, а не стека attention-блоков)
            layer_states = self._per_layer_forward(embedding, tok)
            return logits, layer_states
        return logits

    def _per_layer_forward(self, embedding, tok):
        # Прогоняем каждый LSTM-слой вручную, чтобы иметь доступ к его
        # последовательности скрытых состояний (нужно для RNN-аналога
        # logit-lens энтропийного лосса).
        # То же ограничение cuDNN на eval+backward, что и в forward() выше,
        # плюс эти временные однослойные модули берут веса self.lstm через
        # .data (а не единый непрерывный буфер, как у "настоящего"
        # многослойного nn.LSTM), из-за чего cuDNN предупреждает и
        # пересобирал бы буфер на каждом вызове -- отключение снимает обе
        # проблемы разом.
        h0, c0 = self._init_state(embedding)
        x = tok
        outs = []
        with torch.backends.cudnn.flags(enabled=False):
            for i in range(self.nlayers):
                layer = nn.LSTM(self.d_model, self.d_model, num_layers=1, batch_first=True).to(x.device)
                layer.weight_ih_l0.data = getattr(self.lstm, f"weight_ih_l{i}").data
                layer.weight_hh_l0.data = getattr(self.lstm, f"weight_hh_l{i}").data
                layer.bias_ih_l0.data = getattr(self.lstm, f"bias_ih_l{i}").data
                layer.bias_hh_l0.data = getattr(self.lstm, f"bias_hh_l{i}").data
                x, _ = layer(x, (h0[i:i+1], c0[i:i+1]))
                outs.append(x)
        return outs

    def layer_distributions(self, layer_states):
        dists = []
        for h in layer_states:
            d = F.softmax(self.head(self.ln_f(h)), dim=-1)
            dists.append(d)
        return dists

    @torch.no_grad()
    def greedy_decode(self, embedding, bos_id, eos_id, max_len):
        B = embedding.size(0)
        device = embedding.device
        h, c = self._init_state(embedding)
        tok = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        generated = [tok]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        cur = tok
        for _ in range(max_len - 1):
            emb = self.tok_emb(cur)
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.head(self.ln_f(out[:, -1, :]))
            nxt = logits.argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            generated.append(nxt.unsqueeze(1))
            finished = finished | nxt.eq(eos_id)
            cur = nxt.unsqueeze(1)
            if finished.all():
                break
        return torch.cat(generated, dim=1)


# ---------------------------------------------------------------------------
# Третий, нерекуррентный/безattention-ный EIA: стек causal-дилатированных
# 1D-свёрток (в стиле TCN/WaveNet). Архитектурно независим и от
# TransformerAttacker (attention), и от LSTMAttacker (рекуррентность) --
# используется как вторая нетрансформерная цель в тесте переноса
# Эксперимента 2.
# ---------------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x):
        """x: (B, d_model, T) -> (B, d_model, T), causal (паддинг только
        слева, так что выход в позиции t зависит только от входов на
        позициях <= t)."""
        h = self.act(self.conv(F.pad(x, (self.pad, 0))))
        return x + h  # residual-связь, так как глубина (nlayers) растёт вместе с dilation


class CNNAttacker(nn.Module):
    def __init__(self, vocab_size, emb_dim, d_model=64, nlayers=3, max_len=17, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.emb_proj = nn.Linear(emb_dim, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            CausalConvBlock(d_model, kernel_size, dilation=2 ** i) for i in range(nlayers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.head.weight = self.tok_emb.weight

    def forward(self, embedding, input_ids, return_layer_outputs=False):
        """embedding: (B, d_emb) обуславливает каждую позицию через
        аддитивное смещение (у CNN нет естественного единого слота
        "префикс"/"начальное состояние", как у transformer/LSTM-атакующих)."""
        B, T = input_ids.shape
        ctx = self.emb_proj(embedding).unsqueeze(1)  # (B,1,d), транслируется по T
        pos = self.pos_emb(torch.arange(T, device=input_ids.device)).unsqueeze(0)
        x = self.tok_emb(input_ids) + pos + ctx       # (B,T,d)
        x = x.transpose(1, 2)                          # (B,d,T) для Conv1d

        layer_outs = []
        for blk in self.blocks:
            x = blk(x)
            layer_outs.append(x.transpose(1, 2))        # (B,T,d), для logit-lens

        logits = self.head(self.ln_f(x.transpose(1, 2)))
        if return_layer_outputs:
            return logits, layer_outs
        return logits

    def layer_distributions(self, layer_outs):
        return [F.softmax(self.head(self.ln_f(h)), dim=-1) for h in layer_outs]

    @torch.no_grad()
    def greedy_decode(self, embedding, bos_id, eos_id, max_len):
        B = embedding.size(0)
        device = embedding.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(embedding, generated)
            next_tok = logits[:, -1, :].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished = finished | next_tok.eq(eos_id)
            if finished.all():
                break
        return generated


# ---------------------------------------------------------------------------
# Генератор возмущений EntroGuard G_theta: MLP в стиле DenseNet, R^d -> R^d
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_dim, growth):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, growth)
        self.fc2 = nn.Linear(in_dim + growth, growth)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(torch.cat([x, y1], dim=-1)))
        return torch.cat([x, y1, y2], dim=-1)


class PerturbationGenerator(nn.Module):
    """Компактный аналог 1D-DenseNet G_theta из статьи: R^d -> R^d."""
    def __init__(self, emb_dim, hidden=64, growth=64, n_blocks=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(emb_dim, hidden)
        dim = hidden
        blocks = []
        for _ in range(n_blocks):
            blocks.append(DenseBlock(dim, growth))
            dim = dim + 2 * growth
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, emb_dim)

    def forward(self, e0):
        x = F.relu(self.in_proj(e0))
        for b in self.blocks:
            x = b(x)
        x = self.drop(x)
        return self.out_proj(x)  # e' (ещё без bound-адаптации)
