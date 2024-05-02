from typing import List, Dict, Any

from accelerate import disk_offload
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


app = Flask(__name__)


class RetrievalAugmentor:
    def __init__(self):
        self._default_ef = embedding_functions.DefaultEmbeddingFunction()
        self._chroma_client = chromadb.Client()
        self._collection = self._prepare_db()
        # Loading model directly
        # Handle ValueError: You are trying to offload the whole model to the disk.
        self._tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        disk_offload(model=self._model, offload_dir=".")

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
                "content": "You are a helpful AI assistant and your goal is to "
                "answer questions as ccurately as possible based on the context provided. If you "
                "cannot find the correct answer, reply I don’t know. Be concise and just include "
                "the response.",
            },
            {"role": "user", "content": user_prompt},
        ]
        return self._tokenizer(user_prompt, return_tensors="pt")

    def make_llm_query(self, query: str) -> str:
        generate_ids = self._model.generate(query.input_ids, max_length=600)
        return self._tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


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
    response = response.split("Answer: ", 1)[1].replace("\n", "")
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
