#! /bin/bash

if [ -d ./data ]; then
	echo "Data directory already exist"
else
	mkdir data
fi

if [ -d ./cache ]; then
	echo "Cache directory already exist"
else
	mkdir cache
fi

if [ -d ./test ]; then
	echo "Test directory already exist"
else
	mkdir test
fi

cd neucf
sh init.sh
cd ../

