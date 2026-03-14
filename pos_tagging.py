import nltk
from nltk.corpus import brown
from nltk.tag import DefaultTagger
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def setup():
    """Tải nhanh các gói cần thiết."""
    packages = [
        'brown', 'universal_tagset', 
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng'
    ]
    for p in packages:
        nltk.download(p, quiet=True)

def run_eval(data, name, mode="perceptron"):
    words = [[w for w, t in s] for s in data]
    y_true = [t for s in data for w, t in s]
    
    print(f"-> Đang đánh giá: {name}...")
    
    if mode == "perceptron":
        tagged = nltk.pos_tag_sents(words, tagset="universal")
    else:
        # Baseline simple nhất: mặc định là Danh từ
        tagger = DefaultTagger("NOUN")
        tagged = tagger.tag_sents(words)
        
    y_pred = [t for s in tagged for w, t in s]
    
    report = f"\n>>> KẾT QUẢ: {name}\n" + "-"*50 + "\n"
    report += classification_report(y_true, y_pred, zero_division=0)
    
    scores = (
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    return report, scores

if __name__ == "__main__":
    setup()
    
    # Load data
    sents = brown.tagged_sents(tagset="universal")
    
    # Evaluate
    rep_pt, sc_pt = run_eval(sents, "NLTK Pre-trained (Perceptron)")
    rep_bl, sc_bl = run_eval(sents, "Baseline (Always NOUN)", mode="baseline")
    
    # Summary Table
    summary = "\n" + "="*60 + "\nBẢNG SO SÁNH TỔNG QUAN\n" + "="*60 + "\n"
    summary += f"{'Mô hình':<30} {'Prec':>8} {'Rec':>8} {'F1':>8}\n"
    summary += "-"*60 + "\n"
    summary += f"{'NLTK Perceptron':<30} {sc_pt[0]:>8.4f} {sc_pt[1]:>8.4f} {sc_pt[2]:>8.4f}\n"
    summary += f"{'Baseline (NOUN)':<30} {sc_bl[0]:>8.4f} {sc_bl[1]:>8.4f} {sc_bl[2]:>8.4f}\n"
    summary += "="*60
    
    full_output = rep_pt + rep_bl + summary
    print(full_output)
    
    with open("results_output.txt", "w", encoding="utf-8") as f:
        f.write(full_output)
