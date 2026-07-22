"""CLI diagnostic: dumps aggregated FinBERT sentiment stored in ChromaDB
for a ticker (average score + 5 most recent headlines)."""

from chroma_store import get_news_collection


def analyze_sentiment(ticker):
    print(f"\nConnecting to Vector Database for {ticker}...")
    try:
        collection = get_news_collection()
        results = collection.get(where={"ticker": ticker})

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        total_count = len(documents)

        if total_count == 0:
            print(f" No sentiment data found for {ticker} in the database yet.")
            return

        combined_data = []
        for i in range(total_count):
            combined_data.append({
                "date": metadatas[i].get("date", "Unknown"),
                "score": metadatas[i].get("sentiment_score", 0.5),
                "document": documents[i],
            })
        combined_data.sort(key=lambda x: x["date"], reverse=True)

        scores = [item["score"] for item in combined_data]
        avg_sentiment = sum(scores) / total_count

        print(f" ----- SENTIMENT REPORT: {ticker} -----")
        print(f"Total Documents : {total_count}")
        print(f"Avg Sentiment   : {avg_sentiment:.3f} (0 = Bear, 1 = Bull)")

        print("----- RECENT HEADLINES: -----")
        for record in combined_data[:5]:
            print(f" [{record['date']}] [Score: {record['score']:.3f}] | {record['document']}")

    except Exception as e:
        print(f" DATABASE ERROR: {e}")
        print("Double-check your ChromaDB path and collection name!")


if __name__ == "__main__":
    for target in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]:
        analyze_sentiment(target)
