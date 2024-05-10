#! /usr/bin/bash

query="who is the lead actress in everything everywhere all at once?"
query="who is the director for everything everywhere all at once?"
query="who won the oscar for everything everywhere all at once?"  #
query="who won best picture in 2023 oscars?"  # both correct
query="Tell me something about lady gaga"  # rag better
# query="who won best actor in 2023 oscars?"  # rag better
# query="what did Living - Kazuo Ishiguro get nominated for in the oscars?"  # rag better

echo "Query: $query"

echo "RAG response:"
url="http://127.0.0.1:5000/search?query=$query"
escaped_url=$(echo "$url" | sed 's/ /%20/g')
time curl -s "$escaped_url"

# Collecting performance results
# curl -s -o /dev/null -w "Total time: %{time_total}\n" "$escaped_url"
# curl -s "$escaped_url" && echo "Total time: $(($(date +%s.%N) - $(date +%s.%N))%N)"

# Make the cURL request and capture the output
# response=$(curl -s -w "\nTime taken: %{time_total} seconds\n" -o /dev/null "$escaped_url")

# Print the response
# echo "Response:"
# echo "$response"

# Extract and print the time taken
# time_taken=$(echo "$response" | grep -oP 'Time taken: \K\d+(\.\d+)?')
# echo "Time taken: $time_taken seconds"


echo "Non-RAG response:"
url="http://127.0.0.1:5000/search_simple?query=$query"
escaped_url=$(echo "$url" | sed 's/ /%20/g')
time curl -s "$escaped_url"

# Collecting performance results
# curl -s -o /dev/null -w "Total time: %{time_total}\n" "$escaped_url"
# curl -s "$escaped_url" && echo "Total time: $(($(date +%s.%N) - $(date +%s.%N))%N)"

# Make the cURL request and capture the output
# response=$(curl -s -w "\nTime taken: %{time_total} seconds\n" -o /dev/null "$escaped_url")

# Print the response
# echo "Response:"
# echo "$response"

# Extract and print the time taken
# time_taken=$(echo "$response" | grep -oP 'Time taken: \K\d+(\.\d+)?')
# echo "Time taken: $time_taken seconds"
