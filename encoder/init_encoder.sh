
if [ -d ../cache ]; then
	echo "Cache directory already exist"
else
	mkdir cache
fi

if [ -d ../test ]; then
	echo "Test directory already exist"
else
	mkdir test
fi

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
#vi data/tf_records/train/train_001.tfrecord
#vi data/tf_records/test/test_001.tfrecord
python tf_record_writer.py
python training.py
python training.py


