#! /usr/bin/bash
url="http://127.0.0.1:5000/search?query=who directed everything everywhere all at once?"
escaped_url=$(echo "$url" | sed 's/ /%20/g')
curl "$escaped_url"

