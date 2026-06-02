# Top-writeups archive — playground-series-s6e5

CSV files in this directory are gitignored (`*.csv` in
`.gitignore`). They are derived data; re-fetch any time with the
working-tree CLI:

```bash
mkdir -p audit/writeups-archive
for tid in 703562 703528 703615 703529 703542 703572 703539 703584 703537; do
  kaggle competitions topic-messages playground-series-s6e5 $tid -n -1 -v \
    > audit/writeups-archive/$tid.csv
done
```

The 9 writeups + URL slugs are listed in
`audit/2026-06-02-top-writeups-contrast.md` section G.

To list new writeups posted after this fetch:

```bash
curl -s -H "Authorization: Bearer $KaggleAPIToke" \
  "https://www.kaggle.com/api/v1/competitions/playground-series-s6e5/topics?sortBy=new&pageSize=200" \
  | python3 -c "import sys,json; [print(t['id'],t['title']) \
      for t in json.load(sys.stdin)['topics'] if '/writeups/' in t.get('topicUrl','')]"
```
