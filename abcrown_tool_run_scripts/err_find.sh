for file in logs/*; do
    if ! grep -q "^Result:" "$file"; then
        echo "$file"
    fi
done

