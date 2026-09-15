"""
Два источника данных, выбираются через config.DATA_SOURCE:

  "synthetic"   -- шаблонный корпус, генерируется локально (сеть не
                   нужна). Быстрый офлайн-путь для проверки, что всё
                   работает.
  "personachat" -- реальные персона-фразы PersonaChat (эксперименты по
                   приватности/инверсии) + реальные queries/passages
                   MS MARCO (только метрика полезности retrieval),
                   загружаются через библиотеку Hugging Face `datasets`.
                   См. load_personachat / load_msmarco и NOTES.md "Данные".
"""
import random
import re
from collections import Counter

random.seed(1024)

NAMES = ["alex", "maria", "john", "wei", "fatima", "liam", "sara", "ivan", "noor", "kenji"]
CITIES = ["chicago", "london", "tokyo", "berlin", "cairo", "lagos", "seoul", "lima", "oslo", "delhi"]
JOBS = ["nurse", "teacher", "engineer", "chef", "lawyer", "driver", "artist", "pilot", "farmer", "banker"]
PETS = ["dog", "cat", "parrot", "rabbit", "hamster", "turtle", "fish", "lizard"]
CONDITIONS = ["diabetes", "asthma", "insomnia", "back pain", "migraines", "anxiety", "a knee injury"]
HOBBIES = ["chess", "painting", "hiking", "gaming", "cooking", "cycling", "gardening", "reading", "fishing", "yoga"]
RELATIVES = ["mother", "father", "sister", "brother", "cousin", "grandmother", "uncle", "aunt"]
FOODS = ["sushi", "pizza", "tacos", "curry", "pasta", "dumplings", "salad", "steak"]

TEMPLATES = [
    "i work as a {job} in {city}",
    "my {relative} has {condition} and needs help",
    "i have a pet {pet} named {name}",
    "i enjoy {hobby} on the weekends with {name}",
    "i am planning a trip to {city} next month",
    "i faked my resume to get a job as a {job}",
    "my therapist in {city} knows i struggle with {condition}",
    "what should i cook for dinner tonight , maybe {food}",
    "i just moved to {city} to work as a {job}",
    "my {relative} {name} loves {hobby} and {food}",
    "i think i am developing {condition} from stress at work",
    "can you recommend a good {food} recipe for {name}",
    "i used to live in {city} before becoming a {job}",
    "my {pet} keeps me company while i practice {hobby}",
    "i am saving money to visit my {relative} in {city}",
    "what can you eat when you have {condition}",
    "i told {name} about my {condition} but nobody else knows",
    "i am a {job} who secretly wants to try {hobby}",
    "my {relative} taught me how to cook {food}",
    "i adopted a {pet} after moving to {city}",
]


def _fill(template):
    return template.format(
        name=random.choice(NAMES),
        city=random.choice(CITIES),
        job=random.choice(JOBS),
        pet=random.choice(PETS),
        condition=random.choice(CONDITIONS),
        hobby=random.choice(HOBBIES),
        relative=random.choice(RELATIVES),
        food=random.choice(FOODS),
    )


def build_corpus(n_sentences=900):
    seen = set()
    sentences = []
    tries = 0
    while len(sentences) < n_sentences and tries < n_sentences * 50:
        tries += 1
        t = random.choice(TEMPLATES)
        s = _fill(t)
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    random.shuffle(sentences)
    return sentences


class Vocab:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, sentences, max_size=None):
        """max_size ограничивает словарь самыми частыми словами (остальные
        уходят в UNK) -- нужно, когда в игру вступают реальные корпуса,
        их сырой словарь может доходить до десятков тысяч слов."""
        counts = Counter()
        for s in sentences:
            counts.update(s.split())
        specials = [self.PAD, self.BOS, self.EOS, self.UNK]
        if max_size is not None:
            words = sorted(w for w, _ in counts.most_common(max_size))
        else:
            words = sorted(counts)
        self.itos = specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, max_len):
        ids = [self.bos_id] + [self.stoi.get(w, self.unk_id) for w in sentence.split()] + [self.eos_id]
        ids = ids[:max_len]
        ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.itos[i]
            if w == self.EOS:
                break
            if w in (self.PAD, self.BOS):
                continue
            words.append(w)
        return " ".join(words)


# ---------------------------------------------------------------------------
# Реальные корпуса (Hugging Face `datasets`, скачиваются/кэшируются при
# первом использовании -- см. download_data.py). `datasets` импортируется
# лениво внутри этих функций, чтобы synthetic-путь выше не требовал его
# установки.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:]")


def simple_tokenize(text):
    """Токенизатор слов/пунктуации для реального текста, в нижнем
    регистре. Сокращения ("don't") остаются одним токеном, пунктуация
    отделяется — так что дальше по цепочке обычный str.split() у Vocab
    продолжает работать как валидный токенизатор."""
    return _TOKEN_RE.findall(text.lower().strip())


def _pretokenize(text):
    return " ".join(simple_tokenize(text))


def load_personachat(max_sentences=4000, dataset_name="AlekseyKorshuk/persona-chat", split="train"):
    """Уникальные персона-фразы PersonaChat ("i work as a nurse", "i have
    two cats", ...) -- короткие, однофактные, несущие личную информацию
    предложения — ровно то, что имитировал синтетический корпус для атаки
    инверсии. Первый вызов требует сеть (дальше `datasets` кэширует)."""
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(dataset_name, split=split)
    if "personality" not in ds.column_names:
        raise ValueError(
            f"expected a 'personality' column in {dataset_name!r}, got "
            f"{ds.column_names}. The dataset schema doesn't match what "
            f"load_personachat() assumes -- update it accordingly."
        )

    sentences, seen = [], set()
    for ex in ds:
        for p in ex["personality"]:
            t = _pretokenize(p)
            if t and t not in seen:
                seen.add(t)
                sentences.append(t)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def load_msmarco(max_queries=500, max_passages=5000, dataset_name="microsoft/ms_marco",
                  config_name="v1.1", split="train"):
    """Queries MS MARCO + пул уникальных кандидатных passages, используются
    ТОЛЬКО для метрики полезности retrieval (eq. 5): queries — это то, что
    эмбеддится/защищается, passages — база для retrieval. В обучении
    атакующего не участвуют — там используется только PersonaChat."""
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(dataset_name, config_name, split=split)
    required = {"query", "passages"}
    if not required.issubset(ds.column_names):
        raise ValueError(
            f"expected {required} columns in {dataset_name!r}/{config_name!r}, "
            f"got {ds.column_names}. The dataset schema doesn't match what "
            f"load_msmarco() assumes -- update it accordingly."
        )

    queries, passages, seen = [], [], set()
    for ex in ds:
        if len(queries) < max_queries:
            q = _pretokenize(ex["query"])
            if q:
                queries.append(q)
        if len(passages) < max_passages:
            for p in ex["passages"]["passage_text"]:
                t = _pretokenize(p)
                if t and t not in seen:
                    seen.add(t)
                    passages.append(t)
        if len(queries) >= max_queries and len(passages) >= max_passages:
            break
    return queries, passages


if __name__ == "__main__":
    sents = build_corpus()
    print(f"unique sentences: {len(sents)}")
    v = Vocab(sents)
    print(f"vocab size: {len(v)}")
    for s in sents[:5]:
        print(" -", s)
