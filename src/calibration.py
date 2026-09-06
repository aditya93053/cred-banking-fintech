import json
from .rag import LocalRAG
def main():
    rag = LocalRAG()
    queries = [
      ("loan eligibility", True), ("emi rules", True), ("kyc documents", True),
      ("fraud dispute", True), ("joint account", True), ("weather tomorrow", False),
    ]
    rows=[]
    for q, expected in queries:
        hits=rag.retrieve(q,1)
        rows.append({"query":q,"expected_in_scope":expected,"top1_score":hits[0]["score"] if hits else 0})
    # Deterministic demo threshold: selected between observed in/out values.
    threshold=0.5
    json.dump({"threshold":threshold,"samples":rows}, open("calibration.json","w"), indent=2)
    print(json.dumps({"threshold":threshold,"samples":rows},indent=2))
if __name__=="__main__": main()
