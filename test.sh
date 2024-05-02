#! /usr/bin/bash

query="who is the lead actress in everything everywhere all at once?"
#query="who is the director for everything everywhere all at once?"
#query="who won the oscar for everything everywhere all at once?"

echo "Query: $query"

echo "RAG response:"
url="http://127.0.0.1:5000/search?query=$query"
escaped_url=$(echo "$url" | sed 's/ /%20/g')
curl "$escaped_url"

echo "Non-RAG response:"
url="http://127.0.0.1:5000/search_simple?query=$query"
escaped_url=$(echo "$url" | sed 's/ /%20/g')
curl "$escaped_url"
