if [ -d ./data ]; then
echo "Data directory already exist"
else
mkdir data
fi

if [ -d ./data/tf_records ]; then
echo "Tf_records directory already exist"
else
mkdir data/tf_records
fi

if [ -d ./data/tf_records/train ]; then
echo "train directory already exist"
else
mkdir data/tf_records/train
fi

if [ -d ./data/tf_records/test ]; then
echo "test directory already exist"
else
mkdir data/tf_records/test
fi



