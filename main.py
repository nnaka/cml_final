from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import pipeline


app = Flask(__name__)


class RetrievalAugmentor:
    def __init__(self):
        self._default_ef = embedding_functions.DefaultEmbeddingFunction()
        self._chroma_client = chromadb.Client()
        self._collection = self._prepare_db()
        self._pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def _format_data(self):
        df = pd.read_csv("./data/oscars.csv")
        df = df.loc[df["year_ceremony"] == 2023]
        df = df.dropna(subset=["film"])
        df.loc[:, "category"] = df["category"].str.lower()
        df.loc[:, "text"] = (
            df["name"]
            + " got nominated under the category, "
            + df["category"]
            + ", for the film "
            + df["film"]
            + " to win the award"
        )
        df.loc[df["winner"] == False, "text"] = (
            df["name"]
            + " got nominated under the category, "
            + df["category"]
            + ", for the film "
            + df["film"]
            + " but did not win"
        )
        return df

    def _prepare_db(self):
        collection = self._chroma_client.create_collection(name="my_collection")
        df = self._format_data()
        docs = df["text"].tolist()
        ids = [str(x) for x in df.index.tolist()]
        collection.add(documents=docs, ids=ids)
        print("Completed the uploading of data")
        return collection

    def embed_query(self, query: str, model: Any) -> Any:
        return self._default_ef(query.split())

    def search_chroma_db(self, query: str) -> List[Dict[str, Any]]:
        return self._collection.query(
            query_texts=[query],
            n_results=15,
            include=["documents"],
        )

    def generate_prompt(self, documents: List[Dict[str, Any]]) -> str:
        # Generate a new prompt based on the documents
        return new_prompt

    def generate_augmented_prompt(self, query: str) -> str:
        relevant_documents: List[str] = self.search_chroma_db(query)
        context = "\n".join(str(item) for item in relevant_documents["documents"][0])
        user_prompt = f"""
            Based on the context:
            {context}
            Answer the below query:
            {query}
            """
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": user_prompt},
        ]
        return self._pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def make_llm_query(self, query: str) -> str:
        outputs = self._pipe(
            query,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            top_p=0.95,
        )
        return outputs[0]["generated_text"]


ag: RetrievalAugmentor = RetrievalAugmentor()


@app.route("/add", methods=["PUT"])
def add() -> Any:
    pass


@app.route("/search", methods=["GET"])
def search() -> Any:
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    augmented_prompt: str = ag.generate_augmented_prompt(query)
    print(f"augmented_prompt: {augmented_prompt}")
    response: str = ag.make_llm_query(augmented_prompt)
    # Get the response
    response = response.split("<|assistant|>", 1)[1].replace("\n", "")
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
