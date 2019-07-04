wget https://polybox.ethz.ch/index.php/s/kVAzmLwpDvxujhY/download
mv download ./cache/cache.zip

wget https://polybox.ethz.ch/index.php/s/cpIwvS5tvsdaEpU/download
mv download ./test/test.zip

unzip ./cache/cache.zip
rm ./cache/cache.zip

unzip ./test/test.zip
rm ./test/test.zip