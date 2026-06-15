git init
git add .
git commit -F scripts/git/commit_message.txt
git branch -M main

# Check if remote 'origin' exists before adding
if ! git remote | grep -q 'origin'; then
    echo "Adding new remote origin: git@github.com:Yutong-Feng/MMKG-JEPA-dev.git"
    git remote add origin git@github.com:Yutong-Feng/MMKG-JEPA-dev.git
else
    echo "Remote 'origin' already exists - skipping addition"
fi

git push -u origin main --force
