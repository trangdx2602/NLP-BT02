"""
BT_02: POS Tagging trên Brown Corpus
=====================================
- Bộ tagger 1: N-gram Backoff (Bigram → Unigram → Default)
- Bộ tagger 2: Averaged Perceptron Tagger
- Đánh giá: Precision, Recall, Macro-F1 (sklearn)
"""

import sys
import time
import nltk
from nltk.corpus import brown

class Logger(object):
    def __init__(self, filename="results_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Ghi đồng thời ra Terminal và file text chuẩn UTF-8
sys.stdout = Logger("results_output.txt")

from nltk.tag import UnigramTagger, BigramTagger, DefaultTagger, PerceptronTagger
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# 0. Kiểm tra và tải dữ liệu cần thiết
# ─────────────────────────────────────────────
print("=" * 60)
print("BT_02: POS TAGGING TRÊN BROWN CORPUS")
print("=" * 60)

for pkg in ["brown", "universal_tagset", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

# ─────────────────────────────────────────────
# 1. Tải và chia dữ liệu Brown corpus
# ─────────────────────────────────────────────
print("\n[Phase 1] Tải dữ liệu Brown corpus...")

# Lấy tất cả câu đã gán nhãn, dùng universal tagset (12 nhãn)
tagged_sents = brown.tagged_sents(tagset="universal")
total = len(tagged_sents)

# Chia 80% train, 20% test
split = int(total * 0.8)
train_sents = list(tagged_sents[:split])
test_sents  = list(tagged_sents[split:])

print(f"  Tổng số câu       : {total:,}")
print(f"  Tập huấn luyện    : {len(train_sents):,} câu  (80%)")
print(f"  Tập kiểm thử      : {len(test_sents):,} câu  (20%)")

# Lấy danh sách từ trong test set để predict, và nhãn thật để đánh giá
test_words   = [[word for word, tag in sent] for sent in test_sents]
y_true_flat  = [tag  for sent in test_sents   for word, tag in sent]
print(f"  Tổng số token test: {len(y_true_flat):,}")


# ─────────────────────────────────────────────
# 2. Tagger 1: N-gram Backoff
#    Bigram → Unigram → DefaultTagger("NOUN")
# ─────────────────────────────────────────────
print("\n[Phase 2] Huấn luyện N-gram Backoff Tagger...")
t0 = time.time()

default_tagger = DefaultTagger("NOUN")
unigram_tagger = UnigramTagger(train_sents, backoff=default_tagger)
bigram_tagger  = BigramTagger(train_sents,  backoff=unigram_tagger)

train_time_ngram = time.time() - t0
print(f"  Thời gian train   : {train_time_ngram:.2f}s")

# Predict
t0 = time.time()
ngram_pred_sents = bigram_tagger.tag_sents(test_words)
pred_time_ngram  = time.time() - t0
y_pred_ngram     = [tag for sent in ngram_pred_sents for word, tag in sent]
print(f"  Thời gian predict : {pred_time_ngram:.2f}s")


# ─────────────────────────────────────────────
# 3. Tagger 2: Averaged Perceptron Tagger
# ─────────────────────────────────────────────
print("\n[Phase 3] Huấn luyện Averaged Perceptron Tagger...")
t0 = time.time()

perceptron_tagger = PerceptronTagger(load=False)
perceptron_tagger.train(train_sents)

train_time_pctr = time.time() - t0
print(f"  Thời gian train   : {train_time_pctr:.2f}s")

# Predict
t0 = time.time()
pctr_pred_sents = [perceptron_tagger.tag(sent) for sent in test_words]
pred_time_pctr  = time.time() - t0
y_pred_pctr     = [tag for sent in pctr_pred_sents for word, tag in sent]
print(f"  Thời gian predict : {pred_time_pctr:.2f}s")


# ─────────────────────────────────────────────
# 4. Đánh giá
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Phase 4] ĐÁNH GIÁ KẾT QUẢ")
print("=" * 60)

labels = sorted(set(y_true_flat))  # Các nhãn unique, theo thứ tự abc

# ── Tagger 1: N-gram Backoff ──────────────────
print("\n>>> Tagger 1: N-gram Backoff (Bigram → Unigram → Default)")
print("-" * 60)
report_ngram = classification_report(
    y_true_flat, y_pred_ngram,
    labels=labels,
    zero_division=0
)
print(report_ngram)

# ── Tagger 2: Perceptron ─────────────────────
print(">>> Tagger 2: Averaged Perceptron Tagger")
print("-" * 60)
report_pctr = classification_report(
    y_true_flat, y_pred_pctr,
    labels=labels,
    zero_division=0
)
print(report_pctr)

# ── Bảng tổng hợp so sánh ────────────────────
from sklearn.metrics import f1_score, precision_score, recall_score

def macro_metrics(y_true, y_pred, labels):
    p = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    r = recall_score   (y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f = f1_score       (y_true, y_pred, average="macro", labels=labels, zero_division=0)
    return p, r, f

p1, r1, f1_ngram = macro_metrics(y_true_flat, y_pred_ngram, labels)
p2, r2, f1_pctr  = macro_metrics(y_true_flat, y_pred_pctr,  labels)

print("=" * 60)
print("BẢNG SO SÁNH TỔNG HỢP (Macro Average)")
print("=" * 60)
print(f"{'Tagger':<35} {'Precision':>10} {'Recall':>8} {'Macro-F1':>10}")
print("-" * 60)
print(f"{'N-gram Backoff (Bigram+Unigram)':<35} {p1:>10.4f} {r1:>8.4f} {f1_ngram:>10.4f}")
print(f"{'Averaged Perceptron':<35} {p2:>10.4f} {r2:>8.4f} {f1_pctr:>10.4f}")
print("=" * 60)
print(f"\n=> Mô hình tốt hơn (Macro-F1): ", end="")
if f1_pctr >= f1_ngram:
    print(f"Averaged Perceptron ({f1_pctr:.4f} vs {f1_ngram:.4f})")
else:
    print(f"N-gram Backoff ({f1_ngram:.4f} vs {f1_pctr:.4f})")
